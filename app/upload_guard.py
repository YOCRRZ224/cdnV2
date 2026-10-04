"""Upload safety checks.

Every upload endpoint calls inspect_upload() before the file is stored and
record_upload() after it succeeds. Keeping the logic in this module means the
rules can grow without touching the upload handlers in main.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import httpx
import hmac
import ipaddress
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass

from cryptography.fernet import Fernet
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .storage import _do_upload, HF_REPO_ID, HF_TOKEN

router = APIRouter()

# Video containers that are refused outright, by extension.
BLOCKED_VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".wmv",
    ".flv", ".mpg", ".mpeg", ".3gp", ".3g2", ".ogv",
}

# ISO base media files ("ftyp" box) are used for video AND for images/audio.
# A file is only waved through when its major brand is one of these.
_NON_VIDEO_FTYP_BRANDS = {
    b"avif", b"avis", b"heic", b"heix", b"heim", b"heis",
    b"mif1", b"msf1", b"M4A ", b"M4B ", b"M4P ", b"crx ",
    b"jp2 ", b"jpx ", b"jpm ",
}

LOG_PATH = os.getenv("UPLOAD_LOG_PATH", "/tmp/upload_log.jsonl")
LOG_SALT = os.getenv("UPLOAD_LOG_SALT", "change-me")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
_log_lock = threading.Lock()
# Persistence to the Hugging Face dataset. The dataset is publicly readable, so
# IPs are Fernet-encrypted before upload; without LOG_ENC_KEY nothing is uploaded.
# Generate a key with: python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"
LOG_ENC_KEY = os.getenv("LOG_ENC_KEY", "")
REMOTE_LOG_PATH = "_logs/uploads.jsonl"
SYNC_INTERVAL = int(os.getenv("LOG_SYNC_SECONDS", "60"))
MAX_LOG_ROWS = 50000
_dirty = False
_ready = False
_ready_lock = asyncio.Lock()
_sync_task = None
VPN_BLOCK = os.getenv("VPN_BLOCK", "1") == "1"
VPN_FAIL_OPEN = os.getenv("VPN_FAIL_OPEN", "1") == "1"  # allow upload if lookup fails
PROXYCHECK_KEY = os.getenv("PROXYCHECK_KEY", "")
VPN_ALLOWLIST = {i.strip() for i in os.getenv("VPN_ALLOWLIST", "").split(",") if i.strip()}
_vpn_cache: dict[str, tuple[float, bool | None]] = {}
_VPN_TTL = 6 * 3600
TRUST_CF_CONNECTING_IP = os.getenv("TRUST_CF_CONNECTING_IP", "1") == "1"
TRUSTED_PROXY_HOPS = max(1, int(os.getenv("TRUSTED_PROXY_HOPS", "1")))


def _valid_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def get_client_ip(request: Request) -> str:
    """Best-effort real client IP behind Cloudflare / Render's proxy.

    Order: CF-Connecting-IP, then X-Forwarded-For (counted from the right by
    TRUSTED_PROXY_HOPS so a client cannot simply prepend a fake address), then
    the socket peer.
    """
    if TRUST_CF_CONNECTING_IP:
        ip = _valid_ip(request.headers.get("cf-connecting-ip"))
        if ip:
            return ip

    xff = request.headers.get("x-forwarded-for")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            idx = len(parts) - TRUSTED_PROXY_HOPS
            ip = _valid_ip(parts[idx] if idx >= 0 else parts[0])
            if ip:
                return ip

    return _valid_ip(request.client.host if request.client else None) or "unknown"


def _sniff_video(head: bytes) -> bool:
    """True when the first bytes of a file look like a video container."""
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return head[8:12] not in _NON_VIDEO_FTYP_BRANDS
    if head.startswith(b"\x1a\x45\xdf\xa3"):  # Matroska / WebM
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    if head.startswith(b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"):  # ASF / WMV
        return True
    if head.startswith(b"FLV\x01"):
        return True
    if head[:4] in (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"):  # MPEG
        return True
    if head.startswith(b"OggS") and b"theora" in head[:64]:
        return True
    return False


def is_video_upload(filename: str, temp_path: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    if ext in BLOCKED_VIDEO_EXTENSIONS:
        return True
    try:
        with open(temp_path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return _sniff_video(head)


@dataclass
class UploadContext:
    ip: str
    filename: str
    size: int
    folder: str
    method: str  # "file" or "url"
    user_agent: str = ""


async def inspect_upload(
    request: Request,
    filename: str,
    temp_path: str,
    size: int,
    folder: str,
    method: str = "file",
) -> UploadContext:
    """Run all upload checks. Raises HTTPException when the upload is refused."""
    await _ensure_ready()
    ctx = UploadContext(
        ip=get_client_ip(request),
        filename=filename or "",
        size=size,
        folder=folder,
        method=method,
        user_agent=request.headers.get("user-agent", "")[:300],
    )

    if VPN_BLOCK and ctx.ip not in VPN_ALLOWLIST:
        v = await is_vpn(ctx.ip)
        if v or (v is None and not VPN_FAIL_OPEN):
            _log(ctx, None, "blocked_vpn")
            raise HTTPException(status_code=403, detail="Uploads from VPNs/proxies are not allowed.")

    if is_video_upload(ctx.filename, temp_path):
        _log(ctx, None, "blocked_video")
        raise HTTPException(
            status_code=415,
            detail="Video uploads (including .mp4) are not allowed on this CDN.",
        )

    return ctx


async def is_vpn(ip: str) -> bool | None:
    """True = VPN/proxy/hosting, False = clean, None = lookup failed."""
    try:
        if ipaddress.ip_address(ip).is_private:
            return False
    except ValueError:
        return None
    hit = _vpn_cache.get(ip)
    if hit and hit[0] > time.time():
        return hit[1]
    verdict: bool | None = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"https://proxycheck.io/v2/{ip}", params={"vpn": 1, "key": PROXYCHECK_KEY})
            info = r.json().get(ip, {})
            if "proxy" in info:
                verdict = info["proxy"] == "yes"
    except Exception:
        verdict = None
    if len(_vpn_cache) > 5000:
        _vpn_cache.clear()
    _vpn_cache[ip] = (time.time() + (_VPN_TTL if verdict is not None else 300), verdict)
    return verdict


def _fernet():
    if not LOG_ENC_KEY:
        return None
    try:
        return Fernet(LOG_ENC_KEY.encode())
    except ValueError:
        return None


async def _fetch_remote() -> str | None:
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{REMOTE_LOG_PATH}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as c:
            r = await c.get(url, headers=headers)
        return r.text if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


async def _restore() -> None:
    """After a restart the local file is gone: rebuild it from the HF copy."""
    f = _fernet()
    if not f or os.path.exists(LOG_PATH):
        return
    text = await _fetch_remote()
    if not text:
        return
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
            if row.get("ip"):
                row["ip"] = f.decrypt(row["ip"].encode()).decode()
            out.append(json.dumps(row))
        except Exception:
            continue
    if out:
        with _log_lock, open(LOG_PATH, "a") as fh:
            fh.write("\n".join(out) + "\n")


async def _push() -> None:
    global _dirty
    f = _fernet()
    if not f:
        return
    _dirty = False
    tmp = None
    try:
        with _log_lock, open(LOG_PATH) as fh:
            lines = fh.readlines()[-MAX_LOG_ROWS:]
        out = []
        for l in lines:
            if not l.strip():
                continue
            row = json.loads(l)
            if row.get("ip"):
                row["ip"] = f.encrypt(row["ip"].encode()).decode()
            out.append(json.dumps(row))
        fd, tmp = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(out) + "\n")
        await asyncio.to_thread(_do_upload, tmp, REMOTE_LOG_PATH)
    except Exception:
        _dirty = True  # retry on the next tick
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


async def _sync_loop() -> None:
    while True:
        await asyncio.sleep(SYNC_INTERVAL)
        if _dirty:
            await _push()


async def _ensure_ready() -> None:
    global _ready, _sync_task
    if _ready:
        return
    async with _ready_lock:
        if _ready:
            return
        await _restore()
        if _fernet():
            _sync_task = asyncio.create_task(_sync_loop())
        _ready = True


def _anon(ip: str) -> str:
    return hmac.new(LOG_SALT.encode(), ip.encode(), hashlib.sha256).hexdigest()[:10]


def _log(ctx: UploadContext, hf_path: str | None, status: str) -> None:
    global _dirty
    row = {
        "time": int(time.time()),
        "status": status,
        "filename": ctx.filename,
        "path": hf_path,
        "size": ctx.size,
        "folder": ctx.folder,
        "method": ctx.method,
        "uploader": _anon(ctx.ip),
        "ip": ctx.ip,
        "user_agent": ctx.user_agent,
    }
    try:
        with _log_lock, open(LOG_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        _dirty = True
    except OSError:
        pass


async def record_upload(ctx: UploadContext, hf_path: str) -> None:
    await _ensure_ready()
    _log(ctx, hf_path, "ok")


def _read_log(limit: int) -> list[dict]:
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()[-limit:]
    except OSError:
        return []
    return [json.loads(l) for l in reversed(lines) if l.strip()]


@router.get("/api/uploads")
async def public_upload_log(limit: int = 100):
    """Public log. The uploader is an anonymous ID, never the raw IP."""
    await _ensure_ready()
    rows = _read_log(max(1, min(limit, 500)))
    public = ("time", "status", "filename", "path", "size", "folder", "method", "uploader")
    return JSONResponse({"uploads": [{k: r.get(k) for k in public} for r in rows]})


@router.get("/api/admin/uploads")
async def admin_upload_log(request: Request, limit: int = 200):
    """Full log with raw IPs. Needs: Authorization: Bearer $ADMIN_TOKEN."""
    auth = request.headers.get("authorization", "")
    if not ADMIN_TOKEN or not hmac.compare_digest(auth, f"Bearer {ADMIN_TOKEN}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    await _ensure_ready()
    return JSONResponse({"uploads": _read_log(max(1, min(limit, 2000)))})


_ADMIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>Upload log</title>
<style>
*{box-sizing:border-box}
body{font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;background:#0d1117;color:#e6edf3;max-width:1200px;margin-inline:auto}
h1{font-size:18px;margin:0 0 12px}
form{display:flex;gap:8px;flex-wrap:wrap}
input,select,button{font:inherit;padding:10px 12px;background:#161b22;color:inherit;border:1px solid #30363d;border-radius:8px;min-height:42px}
input{flex:1 1 220px}select{flex:1 1 110px}button{flex:0 0 auto;background:#238636;border-color:#238636;font-weight:600}
#msg{color:#f85149;margin:8px 0 0;min-height:1.4em}#count{color:#8b949e;margin:4px 0 8px;font-size:13px}
a{color:#58a6ff}
table{border-collapse:collapse;width:100%}
th,td{border-bottom:1px solid #30363d;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{color:#8b949e;font-weight:600;font-size:12px;text-transform:uppercase}
.st{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;font-weight:600;background:#23863633;color:#3fb950}
.st.blocked_video,.st.blocked_vpn{background:#f8514933;color:#f85149}
@media(max-width:700px){
  thead{display:none}
  table,tbody,tr,td{display:block;width:100%}
  tr{background:#161b22;border:1px solid #30363d;border-radius:10px;margin:0 0 10px;padding:6px 12px}
  td{border:0;padding:4px 0;display:grid;grid-template-columns:78px 1fr;gap:8px}
  td::before{content:attr(data-l);color:#8b949e;font-size:12px;padding-top:2px}
  td.ua{font-size:12px;color:#8b949e}
}
</style></head><body>
<h1>Upload log</h1>
<form id="f"><input id="t" type="password" placeholder="Admin token" autocomplete="off">
<select id="s"><option value="">All</option><option value="ok">OK</option><option value="blocked_video">Blocked video</option><option value="blocked_vpn">Blocked VPN</option></select>
<button>Load</button></form>
<div id="msg"></div><div id="count"></div>
<table><thead><tr><th>Time</th><th>Status</th><th>File</th><th>Size</th><th>IP</th><th>Uploader</th><th>Via</th><th>User agent</th></tr></thead><tbody id="b"></tbody></table>
<script>
const $=id=>document.getElementById(id);
const H=["Time","Status","File","Size","IP","Uploader","Via","Agent"];
$("t").value=sessionStorage.getItem("tk")||"";
async function load(){
  $("msg").textContent="";
  const r=await fetch("/api/admin/uploads?limit=500",{headers:{Authorization:"Bearer "+$("t").value}});
  if(!r.ok){$("msg").textContent=r.status==401?"Wrong token":"Error "+r.status;return}
  sessionStorage.setItem("tk",$("t").value);
  const f=$("s").value,rows=(await r.json()).uploads.filter(x=>!f||x.status==f),b=$("b");b.textContent="";
  $("count").textContent=rows.length+" entries, newest first";
  for(const x of rows){
    const tr=b.insertRow();
    const v=[new Date(x.time*1000).toLocaleString(),x.status,x.path||x.filename,(x.size/1048576).toFixed(2)+" MB",x.ip,x.uploader,x.method,x.user_agent];
    v.forEach((t,i)=>{
      const td=tr.insertCell();td.dataset.l=H[i];
      if(i==1){const s=document.createElement("span");s.className="st "+x.status;s.textContent=x.status.replace("_"," ");td.append(s)}
      else if(i==2&&x.path){const a=document.createElement("a");a.href="/"+x.path;a.textContent=t;td.append(a)}
      else td.textContent=t;
      if(i==7)td.className="ua";
    });
  }
}
$("f").onsubmit=e=>{e.preventDefault();load()};
if($("t").value)load();
</script></body></html>"""


@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Admin UI. The page itself holds no data; it asks for the token and calls the API."""
    return HTMLResponse(_ADMIN_HTML, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex"})
