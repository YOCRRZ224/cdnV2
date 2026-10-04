"""Upload safety checks.

Every upload endpoint calls inspect_upload() before the file is stored and
record_upload() after it succeeds. Keeping the logic in this module means the
rules can grow without touching the upload handlers in main.py.
"""
from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request

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

    if is_video_upload(ctx.filename, temp_path):
        raise HTTPException(
            status_code=415,
            detail="Video uploads (including .mp4) are not allowed on this CDN.",
        )

    return ctx


async def record_upload(ctx: UploadContext, hf_path: str) -> None:
    """Called after a successful upload. Logging is added in a later commit."""
    return None
  
