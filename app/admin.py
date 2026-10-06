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
from urllib.parse import unquote, urlparse

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


def _search(q: str) -> list:
    """Case-insensitive search over file name, CDN path, IP, GitHub repo/actor, method and status.
    Every word must match. A pasted CDN link works too (only its path is used)."""
    q = (q or "").strip()
    if q.lower().startswith(("http://", "https://")):
        q = unquote(urlparse(q).path).strip("/")
    terms = q.lower().split()
    if not terms:
        return _state["log"]
    out = []
    for r in _state["log"]:
        hay = " ".join((r["name"], r["path"], r["ip"], r["method"], r["status"], r.get("extra", ""))).lower()
        if all(t in hay for t in terms):
            out.append(r)
    return out


@router.get("/api/admin/state")
async def admin_state(request: Request, q: str = ""):
    _require_admin(request)
    await ensure_loaded()
    from .upload_guard import get_client_ip, is_cloudflare_ip, is_public_ip, TRUST_CF_CONNECTING_IP, TRUSTED_PROXY_HOPS
    h = request.headers
    me = get_client_ip(request)
    rows = _search(q)
    return {
        "whoami": {
            "detected": me, "ok": is_public_ip(me) and not is_cloudflare_ip(me),
            "x_forwarded_for": h.get("x-forwarded-for"), "cf_connecting_ip": h.get("cf-connecting-ip"),
            "true_client_ip": h.get("true-client-ip"), "socket_peer": request.client.host if request.client else None,
            "trust_cf_header": TRUST_CF_CONNECTING_IP, "proxy_hops": TRUSTED_PROXY_HOPS,
        },
        "now": int(time.time()), "error": _load_error,
        "log": rows[-300:][::-1], "matched": len(rows), "total": len(_state["log"]),
        "blocked": sum(1 for r in _state["log"] if r["status"] not in ("ok", "deleted")), "bans": _state["bans"],
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
:root{--bg:#0d1117;--sf:#161b22;--bd:#30363d;--ln:#21262d;--tx:#e6edf3;--dim:#9da7b3;--red:#ff7b72;--grn:#56d364;--amb:#d29922;--blu:#58a6ff}
*{box-sizing:border-box}
body{font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0 auto;padding:14px 14px 64px;background:var(--bg);color:var(--tx);max-width:1200px;-webkit-text-size-adjust:100%}
header.bar{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}
h1{font-size:19px;margin:0}h2{font-size:15px;margin:24px 0 8px;color:var(--dim);font-weight:600;scroll-margin-top:12px}
a{color:var(--blu);text-decoration:none}
.links{display:flex;gap:8px;align-items:center}
.tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.tile{display:block;background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:10px 12px;color:var(--tx)}
.tile b{display:block;font-size:21px;line-height:1.2}.tile span{font-size:12px;color:var(--dim)}.tile.amb{border-color:var(--amb)}.tile.amb b{color:var(--amb)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px;border-bottom:1px solid var(--ln);vertical-align:top}th{color:var(--dim);font-weight:600}
.act{white-space:nowrap}.name{overflow-wrap:anywhere}
button{font:inherit;font-size:13px;min-height:34px;padding:5px 12px;border-radius:8px;border:1px solid var(--bd);background:var(--sf);color:var(--tx);cursor:pointer}
button.red{border-color:#da3633;color:var(--red)}button.green{border-color:#238636;color:var(--grn)}button:active{background:var(--ln)}
input{font:inherit;font-size:16px;padding:10px 12px;border-radius:8px;border:1px solid var(--bd);background:var(--bg);color:var(--tx);width:100%;max-width:420px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:break-word}.nw{white-space:nowrap}
.dim{color:var(--dim)}.bad{color:var(--red)}.ok{color:var(--grn)}.warn{color:var(--amb)}
.banner{background:#3d1d1d;border:1px solid #da3633;padding:9px 12px;border-radius:8px;margin-top:10px}
.who{margin-top:22px;background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:10px 12px}.who summary{cursor:pointer}
.who table{margin-top:8px}.who p{margin:8px 0 0;font-size:12px}
#login{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:20px}#login input{flex:1 1 220px}
#qrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px}#qrow input{flex:1 1 260px}
@media(min-width:721px){.tiles{grid-template-columns:repeat(4,1fr)}th,td{padding:8px 10px}}
@media(max-width:720px){
  body{padding:12px 12px 64px}
  table.cards,table.cards tbody{display:block}
  table.cards tr{display:block;background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:10px 12px;margin:0 0 10px}
  table.cards tr:first-child{display:none}
  table.cards td{display:flex;gap:14px;justify-content:space-between;align-items:baseline;border:0;padding:3px 0;text-align:right;overflow-wrap:anywhere}
  table.cards td::before{content:attr(data-l);color:var(--dim);flex:none;text-align:left}
  table.cards td.name{display:block;text-align:left;font-size:14px;font-weight:600;padding-bottom:6px}table.cards td.name::before,table.cards td.act::before{display:none}
  table.cards td:empty{display:none}
  table.cards td.empty{display:block;text-align:left;color:var(--dim)}
  table.cards td.act{display:flex;justify-content:stretch;gap:8px;margin-top:8px;padding-top:10px;border-top:1px solid var(--ln)}
  table.cards td.act button{flex:1;min-height:44px;font-size:14px}
  table.kv tr{display:block;padding:6px 0;border-bottom:1px solid var(--ln)}table.kv td{display:block;border:0;padding:1px 0}table.kv td:first-child{color:var(--dim);font-size:12px}
}
</style></head><body>
<header class="bar"><h1>CDN admin</h1><div class="links" id="nav" style="display:none"><a href="/stats">Stats</a><button onclick="logout()">Log out</button></div></header>
<div id="login"><input id="tok" type="password" placeholder="Admin token" autocomplete="current-password"> <button onclick="go()">Open</button> <span id="lerr" class="bad"></span></div>
<div id="app" style="display:none">
<div id="top"></div>
<h2 id="uploads-h">Recent uploads</h2>
<div id="qrow"><input id="q" type="search" placeholder="Search file name, link, IP, GitHub repo, status..." autocomplete="off"><span id="qinfo" class="dim"></span></div>
<div id="uploads"></div>
<div id="rest"></div>
</div>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=t=>new Date(t*1000).toLocaleString([],{dateStyle:'short',timeStyle:'short'}),nf=n=>Number(n).toLocaleString();
const sz=n=>n>=1048576?(n/1048576).toFixed(1)+' MB':n>=1024?(n/1024).toFixed(1)+' KB':n+' B';
let T=sessionStorage.getItem('t')||'',timer=null,Q='',qt=null;
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
    const s=await api('/api/admin/state?q='+encodeURIComponent(Q));
    document.getElementById('login').style.display='none';document.getElementById('app').style.display='block';document.getElementById('nav').style.display='flex';
    render(s);if(!timer)timer=setInterval(load,20000);
  }catch(e){
    if(first||!T){document.getElementById('lerr').textContent=e.message}
    if(timer){clearInterval(timer);timer=null}
    document.getElementById('login').style.display='flex';document.getElementById('app').style.display='none';document.getElementById('nav').style.display='none';
  }
}
const GH='/api/admin/gh';
let WHO=false;
const cell=(l,v,c)=>`<td data-l="${l}"${c?` class="${c}"`:''}>${v}</td>`;
const empty=(n,t)=>`<tr><td class="empty" colspan="${n}">${t}</td></tr>`;
function logout(){sessionStorage.removeItem('t');T='';if(timer){clearInterval(timer);timer=null}document.getElementById('app').style.display='none';document.getElementById('nav').style.display='none';document.getElementById('login').style.display='flex'}
function render(s){
  const banned=s.bans,nb=Object.keys(banned).length,pend=s.gh.filter(([k,g])=>g.status==='pending'&&g.syncs>=1);
  const tile=(n,l,c,href)=>`<${href?`a href="${href}"`:'div'} class="tile ${c||''}"><b>${n}</b><span>${l}</span></${href?'a':'div'}>`;
  let t=`<div class="tiles">${tile(nf(s.total),'uploads saved')}${tile(nf(s.blocked),'blocked attempts',s.blocked?'amb':'')}${tile(nb,'banned IPs','','#bans')}${tile(pend.length,'awaiting approval',pend.length?'amb':'','#pending')}</div>`;
  if(s.error)t+=`<div class="banner">${esc(s.error)}</div>`;
  if(pend.length){
    t+='<h2 id="pending">Waiting for approval</h2><table class="cards"><tr><th>GitHub account</th><th>Repos</th><th>Syncs</th><th>Last sync</th><th></th></tr>';
    for(const [k,g] of pend)t+=`<tr>${cell('Account',`<a href="https://github.com/${encodeURIComponent(g.owner)}" target="_blank" rel="noopener">${esc(g.owner)}</a>`,'name')}${cell('Repos',esc((g.repos||[]).join(', ')))}${cell('Syncs',g.syncs)}${cell('Last sync',fmt(g.last))}<td class="act">${btn('Allow','green',GH,{key:k,action:'allow'})}${btn('Ban','red',GH,{key:k,action:'ban'},'Ban this GitHub account?')}</td></tr>`;
    t+='</table>';
  }
  document.getElementById('top').innerHTML=t;

  let u='<table class="cards"><tr><th>File</th><th>Time</th><th>IP</th><th>Size</th><th>Via</th><th>Status</th><th></th></tr>';
  for(const r of s.log){
    const st=r.status==='ok'?'ok':r.status==='deleted'?'dim':'bad',live=r.path&&r.status==='ok';
    const acts=(r.method==='github'?'':r.ip in banned?'<span class="dim">IP banned</span>':btn('Ban IP','red','/api/admin/ban',{ip:r.ip},'Ban '+r.ip+'?'))+(live?btn('Delete','red','/api/admin/delete',{path:r.path},'Delete '+r.path+' from the CDN?'):'');
    u+=`<tr>${cell('File',(live?`<a href="/${encodeURI(r.path)}" target="_blank" rel="noopener">${esc(r.name)}</a>`:esc(r.name))+(r.extra?`<div class="dim" style="font-weight:400">${esc(r.extra)}</div>`:''),'name')}${cell('Time',fmt(r.t),'nw')}${cell('IP',esc(r.ip),'mono')}${cell('Size',sz(r.size),'nw')}${cell('Via',esc(r.method))}${cell('Status',esc(r.status),st)}${acts?`<td class="act">${acts}</td>`:'<td class="act" style="display:none"></td>'}</tr>`;
  }
  if(!s.log.length)u+=empty(7,Q?'Nothing matches your search.':'No uploads recorded yet.');
  document.getElementById('uploads').innerHTML=u+'</table>';
  document.getElementById('qinfo').textContent=Q?`${s.matched} match${s.matched===1?'':'es'} of ${s.total}${s.matched>s.log.length?' (showing newest '+s.log.length+')':''}`:`${nf(s.total)} saved`;

  let r='<h2>GitHub accounts</h2><table class="cards"><tr><th>Account</th><th>Status</th><th>Repos</th><th>Syncs</th><th>Last IP</th><th>Last sync</th><th></th></tr>';
  for(const [k,g] of s.gh){const c=g.status==='allowed'?'ok':g.status==='banned'?'bad':'warn';
    r+=`<tr>${cell('Account',esc(g.owner),'name')}${cell('Status',esc(g.status),c)}${cell('Repos',esc((g.repos||[]).join(', ')))}${cell('Syncs',g.syncs)}${cell('Last IP',esc(g.last_ip),'mono')}${cell('Last sync',fmt(g.last))}<td class="act">${g.status!=='allowed'?btn('Allow','green',GH,{key:k,action:'allow'}):btn('Revoke','',GH,{key:k,action:'revoke'})}${g.status!=='banned'?btn('Ban','red',GH,{key:k,action:'ban'},'Ban this GitHub account?'):''}</td></tr>`}
  if(!s.gh.length)r+=empty(7,'No GitHub syncs yet.');
  r+=`</table><p class="dim">Always allowed: ${esc(s.always_allow.join(', ')||'none')}</p><h2 id="bans">Banned IPs</h2><table class="cards"><tr><th>IP</th><th>Since</th><th>Note</th><th></th></tr>`;
  for(const ip in banned)r+=`<tr>${cell('IP',esc(ip),'name mono')}${cell('Since',fmt(banned[ip].t))}${cell('Note',esc(banned[ip].note||''))}<td class="act">${btn('Unban','','/api/admin/unban',{ip})}</td></tr>`;
  if(!nb)r+=empty(4,'None.');
  const w=s.whoami,kv=(a,b)=>`<tr><td>${a}</td><td class="mono">${esc(b==null||b===''?'-':b)}</td></tr>`;
  r+=`</table><details id="who" class="who"${WHO||!w.ok?' open':''}><summary>Your IP as the server sees it: <span class="mono ${w.ok?'ok':'bad'}">${esc(w.detected)}</span> ${w.ok?'&#10003;':'&#10007; looks wrong'}</summary><table class="kv">${kv('X-Forwarded-For',w.x_forwarded_for)}${kv('CF-Connecting-IP',w.cf_connecting_ip)}${kv('Socket peer',w.socket_peer)}${kv('Settings','TRUST_CF_CONNECTING_IP='+(w.trust_cf_header?1:0)+', TRUSTED_PROXY_HOPS='+w.proxy_hops)}</table><p class="dim">This must be your real public IP, or bans will hit the wrong address.</p></details>`;
  document.getElementById('rest').innerHTML=r;
  document.getElementById('who').addEventListener('toggle',e=>{WHO=e.target.open});
}
document.getElementById('q').addEventListener('input',e=>{clearTimeout(qt);qt=setTimeout(()=>{Q=e.target.value.trim();load()},250)});
document.getElementById('tok').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
if(T)load(true);
</script></body></html>"""
