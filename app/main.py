import os
import shutil
import time
import uuid
import io
import json
import zipfile
import tarfile
import tempfile
import asyncio
import socket
import ipaddress
from urllib.parse import urlparse, unquote
import httpx
from collections import deque
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

from .storage import (
    is_file,
    get_file_info,
    list_directory,
    list_files_recursive,
    upload_temp_file,
    upload_folder_scoped,
    repo_stats,
    search_files,
    write_batch_manifest,
    get_batch_manifest,
    HF_BUCKET_ID,
    bucket_url,
    auth_headers,
    format_size,
)
from .gh_oidc import verify_actions_token
from .shortener import shorten_url, get_destination_url
from .upload_guard import get_client_ip, inspect_upload

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

STATIC_DIR = Path(__file__).parent / "static"
_NO_STORE_FILES = {"manifest.json", "sw.js"}


@app.get("/static/{filename}")
async def static_no_cache_root(filename: str):
    if filename not in _NO_STORE_FILES:
        raise HTTPException(status_code=404)
    file_path = STATIC_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        file_path,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/static/icons/{filename}")
async def static_icons(filename: str):
    file_path = STATIC_DIR / "icons" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(
        file_path,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(
        STATIC_DIR / "favicon.ico",
        headers={"Cache-Control": "public, max-age=86400"},
    )


CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn.amit.is-a.dev")
RAW_DOMAIN = os.getenv("RAW_DOMAIN", "raw.cdn.amit.is-a.dev")
RAW_BASE_URL = os.getenv("RAW_BASE_URL", f"https://{RAW_DOMAIN}")
RAW_PREFIX = "raw/"


async def render_context(extra: dict) -> dict:
    stats = await repo_stats()
    ctx = dict(extra)
    ctx["repo_file_count"] = stats["file_count"] if stats else None
    ctx["repo_size_str"] = stats["size_str"] if stats else None
    ctx.setdefault("cdn_base_url", CDN_BASE_URL)
    ctx.setdefault("raw_base_url", RAW_BASE_URL)
    return ctx


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    if request.url.hostname == RAW_DOMAIN:
        return PlainTextResponse("404: File Not Found", status_code=404)
    ctx = await render_context({"page": "404"})
    return templates.TemplateResponse(request, "index.html", ctx, status_code=404)


async def _proxy_stream(client: httpx.AsyncClient, r: httpx.Response):
    try:
        async for chunk in r.aiter_raw():
            yield chunk
    finally:
        await client.aclose()


_WEB_ASSET_MIME_TYPES = {
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".xml": "application/xml",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".wasm": "application/wasm",
    ".webmanifest": "application/manifest+json",
    ".map": "application/json",
}

def _resolve_content_type(filename: str, upstream_content_type: str | None) -> str | None:
    """
    Hugging Face's resolve/ endpoint (and GitHub's raw.githubusercontent.com, for
    that matter) often serves files it doesn't specially recognize back as
    text/plain or application/octet-stream. Combined with X-Content-Type-Options:
    nosniff below, that makes browsers refuse to apply a .css file as a stylesheet
    or run a .js file as a script - the Content-Type has to be right, not just
    close. Override it from the file extension for the asset types this actually
    breaks, rather than trusting whatever the upstream sent.
    """
    ext = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext in (".md", ".txt"):
        return "text/plain; charset=utf-8"
    mime = _WEB_ASSET_MIME_TYPES.get(ext)
    if mime:
        if mime.startswith("text/") or mime in ("application/json", "application/xml", "application/manifest+json"):
            return f"{mime}; charset=utf-8"
        return mime
    return upstream_content_type

async def stream_raw(path: str, request: Request):
    hf_url = bucket_url(path)
    client = httpx.AsyncClient(follow_redirects=True)
    req_headers = dict(auth_headers())
    range_header = request.headers.get("range")
    if range_header:
        req_headers["Range"] = range_header
    req = client.build_request("GET", hf_url, headers=req_headers)
    r = await client.send(req, stream=True)

    if r.status_code not in (200, 206):
        await client.aclose()
        raise HTTPException(status_code=404, detail="File not found")

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "public, max-age=31536000",
        "X-Content-Type-Options": "nosniff",
        "Accept-Ranges": "bytes",
    }

    for h in ["Content-Type", "Content-Encoding", "Content-Length", "Etag", "Content-Range"]:
        if h in r.headers:
            headers[h] = r.headers[h]

    filename = path.split("/")[-1].lower()
    resolved_type = _resolve_content_type(filename, headers.get("Content-Type"))
    if resolved_type:
        headers["Content-Type"] = resolved_type

    return StreamingResponse(
        _proxy_stream(client, r),
        status_code=r.status_code,
        headers=headers,
    )


@app.get("/api/get/{path:path}")
async def download_file(path: str, request: Request):
    hf_url = bucket_url(path)
    client = httpx.AsyncClient(follow_redirects=True)
    req_headers = dict(auth_headers())
    range_header = request.headers.get("range")
    if range_header:
        req_headers["Range"] = range_header
    req = client.build_request("GET", hf_url, headers=req_headers)
    r = await client.send(req, stream=True)

    if r.status_code not in (200, 206):
        await client.aclose()
        raise HTTPException(status_code=404, detail="Not found")

    filename = path.split("/")[-1]
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": _resolve_content_type(filename.lower(), r.headers.get("Content-Type")) or "application/octet-stream",
        "Accept-Ranges": "bytes",
    }

    for h in ["Content-Length", "Content-Encoding", "Etag", "Content-Range"]:
        if h in r.headers:
            headers[h] = r.headers[h]

    return StreamingResponse(
        _proxy_stream(client, r),
        status_code=r.status_code,
        headers=headers,
    )


UPLOAD_LIMIT_PER_MINUTE = 50 * 1024 * 1024
UPLOAD_LIMIT_PER_HOUR = 300 * 1024 * 1024


class _UploadRateLimiter:
    def __init__(self):
        self._usage: dict[str, deque] = {}

    def _prune(self, ip: str, now: float):
        dq = self._usage.get(ip)
        if not dq:
            return
        while dq and now - dq[0][0] > 3600:
            dq.popleft()

    def check_and_record(self, ip: str, size: int):
        now = time.time()
        dq = self._usage.setdefault(ip, deque())
        self._prune(ip, now)
        minute_used = sum(s for t, s in dq if now - t <= 60)
        hour_used = sum(s for t, s in dq)

        if minute_used + size > UPLOAD_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail=f"Upload rate limit exceeded: {format_size(UPLOAD_LIMIT_PER_MINUTE)}/minute. Try again shortly.",
            )

        if hour_used + size > UPLOAD_LIMIT_PER_HOUR:
            raise HTTPException(
                status_code=429,
                detail=f"Upload rate limit exceeded: {format_size(UPLOAD_LIMIT_PER_HOUR)}/hour. Try again later.",
            )

        dq.append((now, size))


_upload_limiter = _UploadRateLimiter()


class _CountRateLimiter:
    """Sliding-window request counter per key (e.g. per client IP)."""

    def __init__(self, per_minute: int, per_hour: int):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self._hits: dict[str, deque] = {}

    def check_and_record(self, key: str):
        now = time.time()
        dq = self._hits.setdefault(key, deque())
        while dq and now - dq[0] > 3600:
            dq.popleft()
        if sum(1 for t in dq if now - t <= 60) >= self.per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.per_minute} requests/minute. Try again shortly.",
            )
        if len(dq) >= self.per_hour:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {self.per_hour} requests/hour. Try again later.",
            )
        dq.append(now)
        if len(self._hits) > 10000:
            self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] <= 3600}


_shorten_limiter = _CountRateLimiter(per_minute=10, per_hour=60)
SHORTEN_MAX_URL_LENGTH = 2048
BATCH_MIN_FILES = 2
ALLOWED_UPLOAD_FOLDERS = {"uploads", "third-party"}


def _validate_folder(folder: str) -> str:
    folder = (folder or "uploads").strip().strip("/")
    if folder not in ALLOWED_UPLOAD_FOLDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid folder. Must be one of: {', '.join(sorted(ALLOWED_UPLOAD_FOLDERS))}.",
        )
    return folder


@app.post("/api/put")
async def handle_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    folder: str = Form("uploads"),
):
    folder = _validate_folder(folder)
    client_ip = get_client_ip(request)
    results = []

    for file in files:
        temp_path = f"/tmp/{uuid.uuid4()}"

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        size = os.path.getsize(temp_path)

        try:
            await inspect_upload(request, file.filename, temp_path, size, folder, "file")
            _upload_limiter.check_and_record(client_ip, size)
        except HTTPException:
            os.remove(temp_path)
            raise

        try:
            hf_path = await upload_temp_file(temp_path, file.filename, folder)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        results.append({
            "filename": file.filename,
            "hf_path": hf_path,
            "size": size,
            "cdn_url": f"{CDN_BASE_URL}/{hf_path}",
            "raw_url": f"{RAW_BASE_URL}/{hf_path}",
        })

    response = {"files": results}

    if len(results) >= BATCH_MIN_FILES:
        batch_id = uuid.uuid4().hex[:10]
        await write_batch_manifest(batch_id, results)
        response["batch_id"] = batch_id
        response["batch_url"] = f"{CDN_BASE_URL}/pack/{batch_id}"

    return response


GH_SYNC_LIMIT_PER_HOUR = 2 * 1024 * 1024 * 1024
_gh_sync_limiter = _UploadRateLimiter()


def _safe_extract_tar(tar: tarfile.TarFile, dest: str):
    dest_real = os.path.realpath(dest)

    for member in tar.getmembers():
        member_path = os.path.realpath(os.path.join(dest, member.name))

        if not member_path.startswith(dest_real + os.sep) and member_path != dest_real:
            raise HTTPException(
                status_code=400,
                detail="Archive contains an unsafe path.",
            )

        if member.issym() or member.islnk():
            raise HTTPException(
                status_code=400,
                detail="Archive contains symlinks, which are not allowed.",
            )

    tar.extractall(dest)


@app.get("/gh-sync")
async def gh_sync_docs(request: Request):
    return templates.TemplateResponse(
        request,
        "gh_sync_docs.html",
        {"page": "gh-sync"},
    )


@app.get("/documentation")
async def documentation(request: Request):
    return templates.TemplateResponse(
        request,
        "documentation.html",
        {"page": "documentation"},
    )


@app.post("/api/gh-sync")
async def gh_sync(
    request: Request,
    token: str = Form(...),
    archive: UploadFile = File(...),
):
    claims = verify_actions_token(token)
    owner, repo = claims["owner"], claims["repo"]
    client_ip = get_client_ip(request)

    contents = await archive.read()
    _gh_sync_limiter.check_and_record(
        f"gh-sync:{owner}/{repo}:{client_ip}",
        len(contents),
    )

    with tempfile.TemporaryDirectory() as tmp_upload, tempfile.TemporaryDirectory() as tmp_extract:
        archive_path = os.path.join(tmp_upload, "payload.tar.gz")

        with open(archive_path, "wb") as f:
            f.write(contents)

        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                _safe_extract_tar(tar, tmp_extract)
        except tarfile.TarError:
            raise HTTPException(
                status_code=400,
                detail="Could not read archive (expected a .tar.gz).",
            )

        dest_prefix = await upload_folder_scoped(
            tmp_extract,
            owner,
            repo,
        )

    return {
        "synced_to": dest_prefix,
        "cdn_url": f"{CDN_BASE_URL}/{dest_prefix}/",
        "raw_url_prefix": f"{RAW_BASE_URL}/{dest_prefix}/",
        "note": "cdn_url is the browsable folder listing. raw_url_prefix isn't a link by itself - append an individual filename to it to get that file's direct raw URL.",
        "ref": claims["ref"],
    }


@app.get("/shorten")
async def url_shortener_page(request: Request):
    ctx = await render_context({"page": "shorten"})
    return templates.TemplateResponse(request, "url_shortener.html", ctx)


class ShortenRequest(BaseModel):
    url: str


@app.post("/api/shorten")
async def api_shorten(request: Request, body: ShortenRequest):
    url = body.url.strip()
    if len(url) > SHORTEN_MAX_URL_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"URL too long (max {SHORTEN_MAX_URL_LENGTH} characters).",
        )
    _shorten_limiter.check_and_record(get_client_ip(request))
    _assert_public_url(url)
    short_id = await shorten_url(url)

    return {
        "id": short_id,
        "masked_url": f"{CDN_BASE_URL}/mask/{short_id}",
    }


@app.get("/mask/{short_id}")
async def mask_redirect(request: Request, short_id: str):
    destination = await get_destination_url(short_id)

    if destination is None:
        raise HTTPException(
            status_code=404,
            detail="Short URL not found",
        )

    return RedirectResponse(
        destination,
        status_code=307,
        headers={"Referrer-Policy": "no-referrer"},
    )


_MAX_URL_UPLOAD_BYTES = UPLOAD_LIMIT_PER_HOUR


def _assert_public_url(url: str):
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise HTTPException(
            status_code=400,
            detail="Only https:// URLs are supported.",
        )

    if not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="Invalid URL.",
        )

    hostname_lower = parsed.hostname.lower()

    if hostname_lower == "localhost" or hostname_lower.endswith(".localhost"):
        raise HTTPException(
            status_code=400,
            detail="URLs pointing to private/internal addresses are not allowed.",
        )

    try:
        addrinfos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve host.",
        )

    for family, _, _, _, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise HTTPException(
                status_code=400,
                detail="URLs pointing to private/internal addresses are not allowed.",
            )


class UploadUrlRequest(BaseModel):
    url: str
    folder: str = "uploads"


@app.post("/api/put-url")
async def handle_upload_from_url(
    request: Request,
    body: UploadUrlRequest,
):
    folder = _validate_folder(body.folder)
    client_ip = get_client_ip(request)
    url = body.url.strip()

    _assert_public_url(url)

    parsed = urlparse(url)
    filename = (
        os.path.basename(unquote(parsed.path))
        or f"download-{uuid.uuid4().hex[:8]}"
    )

    temp_path = f"/tmp/{uuid.uuid4()}"
    downloaded = 0

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        try:
            async with client.stream("GET", url) as r:
                if r.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Source returned status {r.status_code}.",
                    )

                content_length = r.headers.get("Content-Length")

                if content_length and int(content_length) > _MAX_URL_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (limit {format_size(_MAX_URL_UPLOAD_BYTES)} per fetch).",
                    )

                cd = r.headers.get("Content-Disposition", "")

                if "filename=" in cd:
                    filename = cd.split("filename=")[-1].strip('"; ') or filename

                with open(temp_path, "wb") as f:
                    async for chunk in r.aiter_bytes():
                        downloaded += len(chunk)

                        if downloaded > _MAX_URL_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"File too large (limit {format_size(_MAX_URL_UPLOAD_BYTES)} per fetch).",
                            )

                        f.write(chunk)

        except httpx.RequestError:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            raise HTTPException(
                status_code=400,
                detail="Could not fetch the URL.",
            )

        except HTTPException:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    try:
        await inspect_upload(
            request,
            filename,
            temp_path,
            downloaded,
            folder,
            "url",
        )
        _upload_limiter.check_and_record(
            client_ip,
            downloaded,
        )
    except HTTPException:
        os.remove(temp_path)
        raise

    try:
        hf_path = await upload_temp_file(
            temp_path,
            filename,
            folder,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "files": [{
            "filename": filename,
            "hf_path": hf_path,
            "size": downloaded,
            "cdn_url": f"{CDN_BASE_URL}/{hf_path}",
            "raw_url": f"{RAW_BASE_URL}/{hf_path}",
        }]
    }


@app.get("/api/pack-info/{path:path}")
async def zip_stats(path: str):
    clean_path = path.strip("/")

    try:
        files = await list_files_recursive(clean_path)
    except ValueError as e:
        raise HTTPException(
            status_code=413,
            detail=str(e),
        )

    if not files:
        raise HTTPException(
            status_code=404,
            detail="Folder is empty or not found",
        )

    total_size = sum(f["size"] for f in files)

    return {
        "file_count": len(files),
        "total_size": total_size,
        "size_str": format_size(total_size),
    }


@app.get("/api/pack/{path:path}")
async def download_zip(path: str):
    clean_path = path.strip("/")

    try:
        files = await list_files_recursive(clean_path)
    except ValueError as e:
        raise HTTPException(
            status_code=413,
            detail=str(e),
        )

    if not files:
        raise HTTPException(
            status_code=404,
            detail="Folder is empty or not found",
        )

    prefix_len = (
        len(clean_path.rstrip("/")) + 1
        if clean_path
        else 0
    )

    ZIP_FETCH_CONCURRENCY = 8
    semaphore = asyncio.Semaphore(ZIP_FETCH_CONCURRENCY)

    async def fetch(client: httpx.AsyncClient, f: dict):
        hf_url = bucket_url(f["path"])

        async with semaphore:
            r = await client.get(hf_url, headers=auth_headers())

        return f, r

    buffer = io.BytesIO()

    async with httpx.AsyncClient(
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(
            *(fetch(client, f) for f in files)
        )

        with zipfile.ZipFile(
            buffer,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zf:
            for f, r in results:
                if r.status_code != 200:
                    continue

                arcname = (
                    f["path"][prefix_len:]
                    if prefix_len
                    else f["path"]
                )

                zf.writestr(
                    arcname,
                    r.content,
                )

    buffer.seek(0)

    zip_filename = (
        clean_path.rstrip("/").split("/")[-1]
        if clean_path
        else HF_BUCKET_ID.split("/")[-1]
    ) + ".zip"

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{zip_filename}"'
            )
        },
    )


@app.get("/api/find")
async def api_search(q: str = ""):
    results = await search_files(q)
    return {"results": results}


@app.get("/pack/{batch_id}")
async def batch_page(request: Request, batch_id: str):
    manifest = await get_batch_manifest(batch_id)

    if not manifest:
        raise HTTPException(
            status_code=404,
            detail="Batch not found",
        )

    batch_files = []

    for f in manifest.get("files", []):
        item = dict(f)
        item["size_str"] = format_size(
            f.get("size")
        )
        batch_files.append(item)

    ctx = await render_context({
        "page": "batch",
        "batch_id": batch_id,
        "batch_files": batch_files,
    })

    return templates.TemplateResponse(
        request,
        "index.html",
        ctx,
    )


@app.get("/api/pack-info-batch/{batch_id}")
async def zip_stats_batch(batch_id: str):
    manifest = await get_batch_manifest(batch_id)

    if not manifest:
        raise HTTPException(
            status_code=404,
            detail="Batch not found",
        )

    files = manifest.get("files", [])

    if not files:
        raise HTTPException(
            status_code=404,
            detail="Batch is empty",
        )

    total_size = sum(
        f.get("size", 0) or 0
        for f in files
    )

    return {
        "file_count": len(files),
        "total_size": total_size,
        "size_str": format_size(total_size),
    }


@app.get("/api/pack-batch/{batch_id}")
async def download_zip_batch(batch_id: str):
    manifest = await get_batch_manifest(batch_id)

    if not manifest:
        raise HTTPException(
            status_code=404,
            detail="Batch not found",
        )

    files = manifest.get("files", [])

    if not files:
        raise HTTPException(
            status_code=404,
            detail="Batch is empty",
        )

    ZIP_FETCH_CONCURRENCY = 8
    semaphore = asyncio.Semaphore(ZIP_FETCH_CONCURRENCY)

    async def fetch(client: httpx.AsyncClient, f: dict):
        hf_url = bucket_url(f["hf_path"])

        async with semaphore:
            r = await client.get(hf_url, headers=auth_headers())

        return f, r

    seen_names = set()
    buffer = io.BytesIO()

    async with httpx.AsyncClient(
        follow_redirects=True,
    ) as client:
        results = await asyncio.gather(
            *(fetch(client, f) for f in files)
        )

        with zipfile.ZipFile(
            buffer,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zf:
            for f, r in results:
                if r.status_code != 200:
                    continue

                arcname = (
                    f.get("filename")
                    or f["hf_path"].split("/")[-1]
                )

                if arcname in seen_names:
                    arcname = (
                        f"{f['hf_path'].split('/')[-1].split('-')[0]}"
                        f"-{arcname}"
                    )

                seen_names.add(arcname)

                zf.writestr(
                    arcname,
                    r.content,
                )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="batch-{batch_id}.zip"'
            )
        },
    )


@app.api_route(
    "/ping",
    methods=["GET", "HEAD"],
    response_class=PlainTextResponse,
)
async def ping():
    return "Server is awake!"


@app.api_route(
    "/{path:path}",
    methods=["GET", "HEAD"],
)
async def serve(request: Request, path: str):
    clean_path = path.strip("/")

    if request.url.hostname == RAW_DOMAIN:
        if not clean_path:
            return HTMLResponse(
                "Specify a file path.",
                status_code=200,
            )

        try:
            return await stream_raw(
                clean_path,
                request,
            )
        except HTTPException as e:
            if e.status_code == 404:
                entries = await list_directory(clean_path)

                if entries:
                    return RedirectResponse(
                        f"{CDN_BASE_URL}/{clean_path}/"
                    )

            raise

    if (
        clean_path == RAW_PREFIX.rstrip("/")
        or clean_path.startswith(RAW_PREFIX)
    ):
        raw_path = clean_path[len(RAW_PREFIX):]

        if not raw_path:
            return HTMLResponse(
                "/raw/   specify a file path after this prefix.",
                status_code=200,
            )

        return RedirectResponse(
            f"{RAW_BASE_URL}/{raw_path}"
        )

    if clean_path == "new":
        ctx = await render_context({"page": "upload"})
        return templates.TemplateResponse(
            request,
            "index.html",
            ctx,
        )

    if clean_path == "find":
        ctx = await render_context({"page": "search"})
        return templates.TemplateResponse(
            request,
            "index.html",
            ctx,
        )

    if clean_path == "history":
        ctx = await render_context({"page": "history"})
        return templates.TemplateResponse(
            request,
            "index.html",
            ctx,
        )

    if clean_path:
        info = await get_file_info(clean_path)

        if info["exists"]:
            filename = clean_path.split("/")[-1]
            ext = (
                filename.rsplit(".", 1)[-1].lower()
                if "." in filename
                else ""
            )

            ctx = await render_context({
                "page": "file",
                "path": clean_path,
                "filename": filename,
                "raw_base_url": RAW_BASE_URL,
                "file_ext": ext,
                "file_size_str": format_size(info["size"]),
            })

            return templates.TemplateResponse(
                request,
                "index.html",
                ctx,
            )

    items = await list_directory(clean_path)

    if clean_path and not items:
        raise HTTPException(
            status_code=404,
            detail="Not Found",
        )

    extra_ctx = {
        "page": "listing",
        "path": clean_path,
        "items": items,
        "raw_base_url": RAW_BASE_URL,
    }

    ctx = await render_context(extra_ctx)

    return templates.TemplateResponse(
        request,
        "index.html",
        ctx,
    )
