"""Upload safety checks.

Every upload endpoint calls inspect_upload() before the file is stored.
Keeping the logic in this module means the rules can grow without touching
the upload handlers in main.py.
"""
from __future__ import annotations

import httpx
import ipaddress
import os
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from .admin import is_banned, record_blocked


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

VPN_BLOCK = os.getenv("VPN_BLOCK", "1") == "1"
VPN_FAIL_OPEN = os.getenv("VPN_FAIL_OPEN", "1") == "1"  # allow upload if lookup fails
PROXYCHECK_KEY = os.getenv("PROXYCHECK_KEY", "")
VPN_ALLOWLIST = {i.strip() for i in os.getenv("VPN_ALLOWLIST", "").split(",") if i.strip()}
_vpn_cache: dict[str, tuple[float, bool | None]] = {}
_VPN_TTL = 6 * 3600
TRUST_CF_CONNECTING_IP = os.getenv("TRUST_CF_CONNECTING_IP", "0") == "1"
TRUSTED_PROXY_HOPS = max(1, int(os.getenv("TRUSTED_PROXY_HOPS", "1")))


def _valid_ip(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


# Cloudflare's published edge ranges (https://www.cloudflare.com/ips/). A request that
# really arrived through Cloudflare carries the visitor's address in CF-Connecting-IP.
_CLOUDFLARE_NETS = [ipaddress.ip_network(n) for n in (
    "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22", "104.16.0.0/13", "104.24.0.0/14",
    "108.162.192.0/18", "131.0.72.0/22", "141.101.64.0/18", "162.158.0.0/15", "172.64.0.0/13",
    "173.245.48.0/20", "188.114.96.0/20", "190.93.240.0/20", "197.234.240.0/22", "198.41.128.0/17",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32", "2405:8100::/32",
    "2a06:98c0::/29", "2c0f:f248::/32",
)]


def is_cloudflare_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(ip.version == n.version and ip in n for n in _CLOUDFLARE_NETS)


def is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


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
        # Hosts like Render append their own internal (10.x etc.) proxy addresses to
        # the right-hand end. Those are never the visitor, so drop them before counting hops.
        public_parts = list(parts)
        while public_parts and not is_public_ip(public_parts[-1]):
            public_parts.pop()
        parts = public_parts or parts
        # The last remaining entry is the address that actually connected to the platform
        # (appended by it, so the visitor can't forge it). If that is a Cloudflare edge, the
        # request really came through Cloudflare and CF-Connecting-IP is genuine.
        if parts and is_cloudflare_ip(parts[-1]):
            ip = _valid_ip(request.headers.get("cf-connecting-ip"))
            if ip:
                return ip
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

    if await is_banned(ctx.ip):
        await record_blocked(ctx, "banned")
        raise HTTPException(status_code=403, detail="You are banned from uploading to this CDN.")

    if VPN_BLOCK and ctx.ip not in VPN_ALLOWLIST:
        v = await is_vpn(ctx.ip)
        if v or (v is None and not VPN_FAIL_OPEN):
            await record_blocked(ctx, "blocked_vpn")
            raise HTTPException(status_code=403, detail="Uploads from VPNs/proxies are not allowed.")

    if is_video_upload(ctx.filename, temp_path):
        await record_blocked(ctx, "blocked_video")
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
