"""Upload safety checks.

Every upload endpoint calls inspect_upload() before the file is stored and
record_upload() after it succeeds. Keeping the logic in this module means the
rules can grow without touching the upload handlers in main.py.
"""
from __future__ import annotations

import hashlib
import httpx
import hmac
import ipaddress
import json
import os
import threading
import time
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

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


def _anon(ip: str) -> str:
    return hmac.new(LOG_SALT.encode(), ip.encode(), hashlib.sha256).hexdigest()[:10]


def _log(ctx: UploadContext, hf_path: str | None, status: str) -> None:
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
    except OSError:
        pass


async def record_upload(ctx: UploadContext, hf_path: str) -> None:
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
    rows = _read_log(max(1, min(limit, 500)))
    public = ("time", "status", "filename", "path", "size", "folder", "method", "uploader")
    return JSONResponse({"uploads": [{k: r.get(k) for k in public} for r in rows]})


@router.get("/api/admin/uploads")
async def admin_upload_log(request: Request, limit: int = 200):
    """Full log with raw IPs. Needs: Authorization: Bearer $ADMIN_TOKEN."""
    auth = request.headers.get("authorization", "")
    if not ADMIN_TOKEN or not hmac.compare_digest(auth, f"Bearer {ADMIN_TOKEN}"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse({"uploads": _read_log(max(1, min(limit, 2000)))})
