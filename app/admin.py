"""Admin dashboard: who uploaded what, IP bans, and the GitHub-sync allowlist.

State is kept in memory and saved as ONE encrypted blob in the Hugging Face
bucket (_admin/state.enc), so it survives restarts and redeploys. The bucket can
be public, so the blob is Fernet-encrypted; the key comes from ADMIN_DATA_KEY
(preferred) or is derived from ADMIN_TOKEN. With no ADMIN_TOKEN the dashboard,
tracking, bans and the allowlist are all off.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import time
import uuid
from collections import deque

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import storage
from .storage import bucket_url, auth_headers, _do_upload_bytes

router = APIRouter()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
_SECRET = os.getenv("ADMIN_DATA_KEY", "") or ADMIN_TOKEN
GH_ALWAYS_ALLOW = {o.strip().lower() for o in os.getenv("GH_ALWAYS_ALLOW", "notamitgamer").split(",") if o.strip()}
STATE_PATH = "_admin/state.enc"
MAX_LOG_ROWS = 2000
SAVE_INTERVAL = 30

_state: dict = {"log": [], "bans": {}, "gh": {}}
_loaded = False
_retry_at = 0.0
_load_error = ""
_dirty = False
_load_lock = asyncio.Lock()
_save_lock = asyncio.Lock()
_loop_task = None
_auth_fails: dict[str, deque] = {}


def enabled() -> bool:
    return bool(ADMIN_TOKEN)


def _fernet() -> Fernet:
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(_SECRET.encode()).digest()))


# ---------------------------------------------------------------- persistence

async def _fetch() -> dict | None:
    """Saved state, {} if none exists yet, None if it could not be read."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as c:
            r = await c.get(bucket_url(STATE_PATH), headers=auth_headers())
    except httpx.HTTPError:
        return None
    if r.status_code == 404:
        return {}
    if r.status_code != 200:
        return None
    return json.loads(_fernet().decrypt(r.content))


async def ensure_loaded() -> bool:
    global _loaded, _load_error, _loop_task, _retry_at
    if not enabled():
        return False
    if _loaded:
        return True
    if time.time() < _retry_at:
        return False
    async with _load_lock:
        if _loaded:
            return True
        _retry_at = time.time() + 60  # don't hammer the bucket (or stall uploads) while it's failing
        try:
            saved = await _fetch()
        except (InvalidToken, ValueError):
            _load_error = "Saved state could not be decrypted - ADMIN_DATA_KEY/ADMIN_TOKEN changed? Saving is paused so it is not overwritten."
            return False
        if saved is None:
            _load_error = "Could not reach the bucket; running in memory only and retrying."
            return False
        _state["log"] = saved.get("log", []) + _state["log"]
        _state["bans"] = {**saved.get("bans", {}), **_state["bans"]}
        _state["gh"] = {**saved.get("gh", {}), **_state["gh"]}
        _loaded, _load_error, _retry_at = True, "", 0.0
        if _loop_task is None:
            _loop_task = asyncio.create_task(_save_loop())
        return True


async def save() -> bool:
    global _dirty
    if not (enabled() and _loaded):
        return False
    async with _save_lock:
        _dirty = False
        _state["log"] = _state["log"][-MAX_LOG_ROWS:]
        blob = _fernet().encrypt(json.dumps(_state).encode())
        try:
            await asyncio.to_thread(_do_upload_bytes, blob, STATE_PATH)
            return True
        except Exception:
            _dirty = True
            return False


async def _save_loop() -> None:
    while True:
        await asyncio.sleep(SAVE_INTERVAL)
        if _dirty:
            await save()


# ------------------------------------------------------------------ recording

def _row(ip: str, name: str, path: str, size: int, method: str, ua: str = "", status: str = "ok", extra: str = "") -> None:
    global _dirty
    _state["log"].append({
        "id": uuid.uuid4().hex[:8], "t": int(time.time()), "ip": ip, "name": (name or "")[:200],
        "path": path, "size": size, "method": method, "ua": (ua or "")[:200], "status": status, "extra": extra[:100],
    })
    _dirty = True


async def record_upload(ctx, path: str) -> None:
    if await ensure_loaded():
        _row(ctx.ip, ctx.filename, path, ctx.size, ctx.method, ctx.user_agent)


async def record_blocked(ctx, reason: str) -> None:
    if await ensure_loaded():
        _row(ctx.ip, ctx.filename, "", ctx.size, ctx.method, ctx.user_agent, status=reason)


async def is_banned(ip: str) -> bool:
    return enabled() and await ensure_loaded() and ip in _state["bans"]


# ------------------------------------------------------- GitHub sync allowlist

def _gh_key(owner: str, owner_id: str) -> str:
    return owner_id or owner.lower()


async def gh_check(owner: str, owner_id: str) -> None:
    """Raise unless this GitHub account may sync. A new account gets one free sync."""
    if not await ensure_loaded():
        return
    rec = _state["gh"].get(_gh_key(owner, owner_id))
    if rec and rec["status"] == "banned":
        raise HTTPException(status_code=403, detail="This GitHub account is banned from the CDN.")
    if owner.lower() in GH_ALWAYS_ALLOW or not rec or rec["status"] == "allowed":
        return
    if rec["syncs"] >= 1:
        raise HTTPException(
            status_code=403,
            detail=f"First sync from '{owner}' was accepted; further syncs need approval from the CDN owner.",
        )


async def gh_record(owner: str, owner_id: str, repo: str, actor: str, ip: str, size: int, dest: str) -> None:
    global _dirty
    if not await ensure_loaded():
        return
    now = int(time.time())
    key = _gh_key(owner, owner_id)
    status = "allowed" if owner.lower() in GH_ALWAYS_ALLOW else "pending"
    rec = _state["gh"].setdefault(key, {"status": status, "first": now, "syncs": 0, "repos": []})
    rec.update(owner=owner, owner_id=owner_id, last=now, last_ip=ip, last_actor=actor)
    rec["syncs"] += 1
    if repo not in rec["repos"]:
        rec["repos"] = (rec["repos"] + [repo])[-20:]
    _row(ip, f"{owner}/{repo}", dest, size, "github", status="ok", extra=f"actor {actor}")
    _dirty = True
    if rec["syncs"] == 1:
        await save()  # new account: persist right away so it shows up for approval


# ------------------------------------------------------------------ dashboard

def _require_admin(request: Request) -> None:
    if not enabled():
        raise HTTPException(status_code=404, detail="Not found")
    from .upload_guard import get_client_ip  # lazy: upload_guard imports this module
    ip = get_client_ip(request)
    now = time.time()
    fails = _auth_fails.setdefault(ip, deque())
    while fails and now - fails[0] > 60:
        fails.popleft()
    if len(fails) >= 8:
        raise HTTPException(status_code=429, detail="Too many attempts. Wait a minute.")
    if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {ADMIN_TOKEN}"):
        fails.append(now)
        raise HTTPException(status_code=401, detail="Unauthorized")


class IpBody(BaseModel):
    ip: str
    note: str = ""


class GhBody(BaseModel):
    key: str
    action: str  # allow | revoke | ban


class PathBody(BaseModel):
    path: str


@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    if not enabled():
        raise HTTPException(status_code=404, detail="Not found")
    return HTMLResponse(_PAGE, headers={"Cache-Control": "no-store", "X-Robots-Tag": "noindex"})


@router.get("/api/admin/state")
async def admin_state(request: Request):
    _require_admin(request)
    await ensure_loaded()
    from .upload_guard import get_client_ip, is_cloudflare_ip, is_public_ip, TRUST_CF_CONNECTING_IP, TRUSTED_PROXY_HOPS
    h = request.headers
    me = get_client_ip(request)
    return {
        "whoami": {
            "detected": me, "ok": is_public_ip(me) and not is_cloudflare_ip(me),
            "x_forwarded_for": h.get("x-forwarded-for"), "cf_connecting_ip": h.get("cf-connecting-ip"),
            "true_client_ip": h.get("true-client-ip"), "socket_peer": request.client.host if request.client else None,
            "trust_cf_header": TRUST_CF_CONNECTING_IP, "proxy_hops": TRUSTED_PROXY_HOPS,
        },
        "now": int(time.time()), "error": _load_error,
        "log": _state["log"][-300:][::-1], "bans": _state["bans"],
        "gh": sorted(_state["gh"].items(), key=lambda kv: kv[1].get("last", 0), reverse=True),
        "always_allow": sorted(GH_ALWAYS_ALLOW),
    }


async def _mutate_and_save() -> dict:
    if not await save():
        raise HTTPException(status_code=503, detail=_load_error or "Could not save to the bucket - try again.")
    return {"ok": True}


@router.post("/api/admin/ban")
async def admin_ban(request: Request, body: IpBody):
    _require_admin(request)
    if not await ensure_loaded():
        raise HTTPException(status_code=503, detail=_load_error)
    try:
        ip = str(ipaddress.ip_address(body.ip.strip()))
    except ValueError:  # also stops "unknown" (undetectable IP) from banning everyone
        raise HTTPException(status_code=400, detail="Not a valid IP address.")
    from .upload_guard import is_cloudflare_ip
    if not ipaddress.ip_address(ip).is_global or is_cloudflare_ip(ip):
        raise HTTPException(
            status_code=400,
            detail="That is an internal or Cloudflare address, not a real visitor (banning it could block many people). The server is not seeing visitor IPs correctly - see 'How the server sees you' at the top of the dashboard.",
        )
    _state["bans"][ip] = {"t": int(time.time()), "note": body.note[:100]}
    return await _mutate_and_save()


@router.post("/api/admin/unban")
async def admin_unban(request: Request, body: IpBody):
    _require_admin(request)
    if not await ensure_loaded():
        raise HTTPException(status_code=503, detail=_load_error)
    _state["bans"].pop(body.ip.strip(), None)
    return await _mutate_and_save()


@router.post("/api/admin/gh")
async def admin_gh(request: Request, body: GhBody):
    _require_admin(request)
    if not await ensure_loaded():
        raise HTTPException(status_code=503, detail=_load_error)
    rec = _state["gh"].get(body.key)
    if not rec or body.action not in {"allow", "revoke", "ban"}:
        raise HTTPException(status_code=400, detail="Unknown account or action.")
    rec["status"] = {"allow": "allowed", "revoke": "pending", "ban": "banned"}[body.action]
    return await _mutate_and_save()


@router.post("/api/admin/delete")
async def admin_delete(request: Request, body: PathBody):
    _require_admin(request)
    if not await ensure_loaded():
        raise HTTPException(status_code=503, detail=_load_error)
    path = body.path.strip().strip("/")
    parts = path.split("/")
    if len(parts) < 2 or ".." in parts or path.startswith("_"):
        raise HTTPException(status_code=400, detail="Invalid path.")
    await asyncio.to_thread(storage.delete_object, path)
    for r in _state["log"]:
        if r["path"] == path:
            r["status"] = "deleted"
    return await _mutate_and_save()


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>CDN admin</title>
<style>
*{box-sizing:border-box}body{font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;background:#0d1117;color:#e6edf3;max-width:1300px;margin-inline:auto}
h1{font-size:18px;margin:0 0 12px}h2{font-size:15px;margin:22px 0 8px;color:#9da7b3}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #21262d;vertical-align:top}
th{color:#9da7b3;font-weight:600}.wrap{overflow-x:auto}
button{font:inherit;font-size:12px;padding:3px 9px;border-radius:6px;border:1px solid #30363d;background:#161b22;color:#e6edf3;cursor:pointer;margin-right:4px}
button.red{border-color:#da3633;color:#ff7b72}button.green{border-color:#238636;color:#56d364}button:hover{background:#21262d}
input{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;width:min(360px,100%)}
.mono{font-family:ui-monospace,monospace;font-size:12px}.dim{color:#9da7b3}.bad{color:#ff7b72}.ok{color:#56d364}.warn{color:#d29922}
.banner{background:#3d1d1d;border:1px solid #da3633;padding:8px 12px;border-radius:6px;margin-bottom:12px}
a{color:#58a6ff;text-decoration:none}
</style></head><body>
<h1>CDN admin</h1>
<div id="login"><input id="tok" type="password" placeholder="Admin token" autocomplete="off"> <button onclick="go()">Open</button> <span id="lerr" class="bad"></span></div>
<div id="app" style="display:none"></div>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=t=>new Date(t*1000).toLocaleString();
const sz=n=>n>=1048576?(n/1048576).toFixed(1)+' MB':n>=1024?(n/1024).toFixed(1)+' KB':n+' B';
let T=sessionStorage.getItem('t')||'',timer=null;
document.getElementById('tok').value='';
async function api(p,body){
  const r=await fetch(p,{method:body?'POST':'GET',headers:{Authorization:'Bearer '+T,'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  if(r.status===401||r.status===429){sessionStorage.removeItem('t');throw new Error((await r.json()).detail)}
  const j=await r.json();if(!r.ok)throw new Error(j.detail||r.status);return j;
}
function go(){T=document.getElementById('tok').value.trim();sessionStorage.setItem('t',T);load(true)}
const btn=(label,cls,path,body,ask)=>`<button class="${cls}" data-p="${esc(path)}" data-b="${esc(JSON.stringify(body))}" data-ask="${esc(ask||'')}">${esc(label)}</button>`;
document.addEventListener('click',async e=>{
  const b=e.target.closest('button[data-p]');if(!b)return;
  if(b.dataset.ask&&!confirm(b.dataset.ask))return;
  try{await api(b.dataset.p,JSON.parse(b.dataset.b));load()}catch(err){alert(err.message)}
});
async function load(first){
  try{
    const s=await api('/api/admin/state');
    document.getElementById('login').style.display='none';document.getElementById('app').style.display='block';
    render(s);if(!timer)timer=setInterval(load,20000);
  }catch(e){
    if(first||!T){document.getElementById('lerr').textContent=e.message}
    if(timer){clearInterval(timer);timer=null}
    document.getElementById('login').style.display='block';document.getElementById('app').style.display='none';
  }
}
const GH='/api/admin/gh';
function render(s){
  const banned=s.bans,pending=s.gh.filter(([k,g])=>g.status==='pending'&&g.syncs>=1);
  let h='';
  if(s.error)h+=`<div class="banner">${esc(s.error)}</div>`;
  const w=s.whoami;
  h+=`<h2>How the server sees you</h2><div class="wrap"><table>
    <tr><td>Detected IP (used for bans)</td><td class="mono ${w.ok?'ok':'bad'}">${esc(w.detected)}</td></tr>
    <tr><td>X-Forwarded-For</td><td class="mono">${esc(w.x_forwarded_for||'-')}</td></tr>
    <tr><td>CF-Connecting-IP</td><td class="mono">${esc(w.cf_connecting_ip||'-')}</td></tr>
    <tr><td>Socket peer</td><td class="mono">${esc(w.socket_peer||'-')}</td></tr>
    <tr><td>Settings</td><td class="mono">TRUST_CF_CONNECTING_IP=${w.trust_cf_header?1:0}, TRUSTED_PROXY_HOPS=${w.proxy_hops}</td></tr></table></div>
    <p class="dim">The detected IP must be YOUR real public IP. If it is not, bans will hit the wrong address.</p>`;
  if(pending.length){
    h+='<h2>Waiting for approval</h2><div class="wrap"><table><tr><th>GitHub account</th><th>Repos</th><th>Syncs</th><th>Last sync</th><th></th></tr>';
    for(const [k,g] of pending)h+=`<tr><td><a href="https://github.com/${encodeURIComponent(g.owner)}" target="_blank" rel="noopener">${esc(g.owner)}</a></td><td>${esc((g.repos||[]).join(', '))}</td><td>${g.syncs}</td><td>${fmt(g.last)}</td>
      <td>${btn('Allow','green',GH,{key:k,action:'allow'})}${btn('Ban','red',GH,{key:k,action:'ban'},'Ban this GitHub account?')}</td></tr>`;
    h+='</table></div>';
  }
  h+='<h2>GitHub accounts</h2><div class="wrap"><table><tr><th>Account</th><th>Status</th><th>Repos</th><th>Syncs</th><th>Last IP</th><th>Last sync</th><th></th></tr>';
  for(const [k,g] of s.gh){const c=g.status==='allowed'?'ok':g.status==='banned'?'bad':'warn';
    h+=`<tr><td>${esc(g.owner)}</td><td class="${c}">${esc(g.status)}</td><td>${esc((g.repos||[]).join(', '))}</td><td>${g.syncs}</td><td class="mono">${esc(g.last_ip)}</td><td>${fmt(g.last)}</td>
    <td>${g.status!=='allowed'?btn('Allow','green',GH,{key:k,action:'allow'}):btn('Revoke','',GH,{key:k,action:'revoke'})}${g.status!=='banned'?btn('Ban','red',GH,{key:k,action:'ban'},'Ban this GitHub account?'):''}</td></tr>`}
  if(!s.gh.length)h+='<tr><td colspan="7" class="dim">No GitHub syncs yet.</td></tr>';
  h+=`</table></div><p class="dim">Always allowed: ${esc(s.always_allow.join(', ')||'none')}</p>`;
  h+='<h2>Banned IPs</h2><div class="wrap"><table><tr><th>IP</th><th>Since</th><th>Note</th><th></th></tr>';
  for(const ip in banned)h+=`<tr><td class="mono">${esc(ip)}</td><td>${fmt(banned[ip].t)}</td><td>${esc(banned[ip].note)}</td><td>${btn('Unban','','/api/admin/unban',{ip})}</td></tr>`;
  if(!Object.keys(banned).length)h+='<tr><td colspan="4" class="dim">None.</td></tr>';
  h+='</table></div><h2>Recent uploads</h2><div class="wrap"><table><tr><th>Time</th><th>IP</th><th>File</th><th>Size</th><th>Via</th><th>Status</th><th></th></tr>';
  for(const r of s.log){
    const st=r.status==='ok'?'ok':r.status==='deleted'?'dim':'bad',live=r.path&&r.status==='ok';
    h+=`<tr><td>${fmt(r.t)}</td><td class="mono" title="${esc(r.ua)}">${esc(r.ip)}</td>
    <td>${live?`<a href="/${encodeURI(r.path)}" target="_blank" rel="noopener">${esc(r.name)}</a>`:esc(r.name)}${r.extra?`<div class="dim">${esc(r.extra)}</div>`:''}</td>
    <td>${sz(r.size)}</td><td>${esc(r.method)}</td><td class="${st}">${esc(r.status)}</td>
    <td>${r.method==='github'?'':r.ip in banned?'<span class="dim">banned</span>':btn('Ban IP','red','/api/admin/ban',{ip:r.ip},'Ban '+r.ip+'?')}${live?btn('Delete','red','/api/admin/delete',{path:r.path},'Delete '+r.path+' from the CDN?'):''}</td></tr>`;
  }
  if(!s.log.length)h+='<tr><td colspan="7" class="dim">No uploads recorded yet.</td></tr>';
  h+='</table></div>';
  document.getElementById('app').innerHTML=h;
}
if(T)load(true);
</script></body></html>"""
