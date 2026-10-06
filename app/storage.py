import os
import re
import time
import unicodedata
import uuid
import json
import tempfile
import asyncio
import httpx
from huggingface_hub import HfApi

# Files live in a Hugging Face Storage Bucket (mutable, non-versioned object
# storage) instead of a git-backed dataset repo: every write is a plain object
# write rather than a commit, so uploads and folder syncs avoid git overhead and
# commit rate limits. HF_REPO_ID is still honoured as a fallback so existing
# deployments keep working until HF_BUCKET_ID is set.
HF_BUCKET_ID = os.getenv("HF_BUCKET_ID") or os.getenv("HF_REPO_ID", "notamitgamer/cdn")
HF_REPO_ID = HF_BUCKET_ID  # backwards-compatible alias used by older imports
HF_TOKEN = os.getenv("HF_TOKEN")
BUCKET_PRIVATE = os.getenv("HF_BUCKET_PRIVATE", "0") == "1"
CACHE_TTL = 60
MAX_CACHE_SIZE = 500
ZIP_MAX_FILES = 300
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024
REPO_STATS_CACHE_TTL = 300
SEARCH_RESULT_LIMIT = 10

api = HfApi(token=HF_TOKEN)


def bucket_url(path: str) -> str:
    """Direct (CDN-backed) download URL for an object in the bucket."""
    return f"https://huggingface.co/buckets/{HF_BUCKET_ID}/resolve/{path}"


def auth_headers() -> dict:
    """Bearer header, only needed when the bucket is private."""
    if BUCKET_PRIVATE and HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}


_bucket_ready = False


def _ensure_bucket():
    global _bucket_ready
    if _bucket_ready:
        return
    api.create_bucket(HF_BUCKET_ID, private=BUCKET_PRIVATE, exist_ok=True)
    _bucket_ready = True

_cache = {}

_key_locks: dict[str, asyncio.Lock] = {}

def _get_key_lock(key: str) -> asyncio.Lock:
    lock = _key_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _key_locks[key] = lock
        
        if len(_key_locks) > MAX_CACHE_SIZE:
            _key_locks.pop(next(iter(_key_locks)))
    return lock

def _get_cache(key):
    if key in _cache:
        expiry, value = _cache[key]
        if expiry > time.time():
            _cache[key] = _cache.pop(key)
            return value
        del _cache[key]
    return None

def _set_cache(key, value):
    if key in _cache:
        del _cache[key]
    elif len(_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]

    _cache[key] = (time.time() + CACHE_TTL, value)

async def is_file(path: str) -> bool:
    info = await get_file_info(path)
    return info["exists"]

async def get_file_info(path: str) -> dict:
    if not path:
        return {"exists": False, "size": None, "content_type": None}

    cache_key = f"file_info_{path}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    async with _get_key_lock(cache_key):
        cached = _get_cache(cache_key)
        if cached is not None:
            return cached

        hf_url = bucket_url(path)
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.head(hf_url, headers=auth_headers())
            exists = r.status_code == 200
            size = None
            content_type = None
            if exists:
                try:
                    size = int(r.headers["Content-Length"]) if "Content-Length" in r.headers else None
                except (ValueError, TypeError):
                    size = None
                content_type = r.headers.get("Content-Type")
            result = {"exists": exists, "size": size, "content_type": content_type}
            _set_cache(cache_key, result)
            return result

def _under(path: str, items: list) -> list:
    """Keep only entries inside `path` (defensive: guards against prefix over-match
    such as 'docs' also matching 'docs-old/...')."""
    if not path:
        return items
    prefix = path.rstrip("/") + "/"
    return [it for it in items if it.path.startswith(prefix)]

def _fetch_tree(path: str):
    path = path.strip("/")
    items = list(api.list_bucket_tree(
        HF_BUCKET_ID, prefix=path or None, recursive=False
    ))
    return _under(path, items)

def _fetch_tree_recursive(path: str = ""):
    path = path.strip("/")
    items = list(api.list_bucket_tree(
        HF_BUCKET_ID, prefix=path or None, recursive=True
    ))
    return _under(path, items)

_HIDDEN_PREFIXES = ("_batches", "_shortened", "_logs", "_admin",)

def _is_hidden_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in _HIDDEN_PREFIXES)

async def list_directory(path: str):
    cache_key = f"list_dir_{path}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    try:
        items = await asyncio.to_thread(_fetch_tree, path)
    except Exception as e:
        print(f"[storage] list_repo_tree failed for {path!r}: {e}")
        return []
    
    files_and_folders = []
    for item in items:
        if _is_hidden_path(item.path):
            continue

        name = item.path.split("/")[-1]
        is_dir = not hasattr(item, "size")
        size = getattr(item, "size", 0)
        
        size_str = format_size(size) if not is_dir else "-"

        files_and_folders.append({
            "name": name,
            "path": item.path,
            "is_dir": is_dir,
            "size_str": size_str
        })
    
    files_and_folders.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    
    _set_cache(cache_key, files_and_folders)
    return files_and_folders

async def list_files_recursive(path: str):
    try:
        items = await asyncio.to_thread(_fetch_tree_recursive, path)
    except Exception:
        return []

    files = []
    total_size = 0
    for item in items:
        if _is_hidden_path(item.path):
            continue
        if hasattr(item, "size"):
            size = getattr(item, "size", 0) or 0
            total_size += size
            files.append({"path": item.path, "size": size})
            if len(files) > ZIP_MAX_FILES or total_size > ZIP_MAX_TOTAL_BYTES:
                raise ValueError(
                    f"Folder too large to zip (limit: {ZIP_MAX_FILES} files / "
                    f"{ZIP_MAX_TOTAL_BYTES // (1024 * 1024)} MB)."
                )
    return files

def format_size(size_bytes) -> str:
    if size_bytes is None:
        return "-"
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.2f} {units[idx]}"

async def _get_full_tree():

    cache_key = "full_tree"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    async with _get_key_lock(cache_key):
        cached = _get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            raw_items = await asyncio.to_thread(_fetch_tree_recursive, "")
        except Exception as e:
            print(f"[storage] full tree fetch failed: {e}")
            return []

        items = []
        for item in raw_items:
            if _is_hidden_path(item.path):
                continue

            name = item.path.split("/")[-1]
            is_dir = not hasattr(item, "size")
            size = getattr(item, "size", 0) or 0
            up = getattr(item, "uploaded_at", None)
            items.append({
                "name": name,
                "path": item.path,
                "is_dir": is_dir,
                "size": size,
                "ts": int(up.timestamp()) if up else 0,
                "size_str": format_size(size) if not is_dir else "-",
            })

        _set_cache(cache_key, items)
        return items

async def repo_stats():
    cache_key = "repo_stats"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    items = await _get_full_tree()
    if not items:
        return None

    file_count = sum(1 for it in items if not it["is_dir"])
    total_size = sum(it["size"] for it in items if not it["is_dir"])

    result = {"file_count": file_count, "size_str": format_size(total_size)}
    _set_cache(cache_key, result)
    return result

async def search_files(query: str):
    q = query.strip().lower()
    if not q:
        return []

    items = await _get_full_tree()
    results = [
        it for it in items
        if q in it["name"].lower() and it["name"] != ".gitattributes"
    ]
    results.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
    return [
        {"name": it["name"], "path": it["path"], "is_dir": it["is_dir"], "size_str": it["size_str"]}
        for it in results[:SEARCH_RESULT_LIMIT]
    ]

def _do_upload(temp_path: str, hf_path: str):
    _ensure_bucket()
    api.batch_bucket_files(
        HF_BUCKET_ID,
        add=[(temp_path, hf_path)],
    )

def _do_upload_bytes(data: bytes, hf_path: str):
    _ensure_bucket()
    api.batch_bucket_files(
        HF_BUCKET_ID,
        add=[(data, hf_path)],
    )

def delete_object(path: str):
    """Delete one object, or everything under path/ (for a synced owner/repo folder)."""
    _ensure_bucket()
    path = path.strip("/")
    items = [it.path for it in api.list_bucket_tree(HF_BUCKET_ID, prefix=path, recursive=True)
             if it.type == "file" and (it.path == path or it.path.startswith(path + "/"))]
    if items:
        api.batch_bucket_files(HF_BUCKET_ID, delete=items)
    _cache.clear()


_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str, max_len: int = 100) -> str:
    """Turn any user-supplied filename into a safe object-key segment.

    Accepts spaces, accents, symbols, emoji and path fragments: directory parts
    are dropped, accents are folded to ASCII, and anything else unsafe becomes
    a single underscore. The extension is kept.
    """
    name = (filename or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = _UNSAFE_NAME_CHARS.sub("_", stem).strip("._") or "file"
    ext = _UNSAFE_NAME_CHARS.sub("", ext)[:10]
    stem = stem[: max_len - len(ext) - 1]
    return f"{stem}.{ext}" if ext else stem


async def upload_temp_file(temp_path: str, filename: str, folder: str = "uploads") -> str:
    slug = str(uuid.uuid4())[:8]
    safe_filename = sanitize_filename(filename)
    hf_path = f"{folder}/{slug}-{safe_filename}"

    await asyncio.to_thread(_do_upload, temp_path, hf_path)

    _cache.clear()
    return hf_path

def _do_upload_folder_scoped(local_dir: str, dest_prefix: str):
    # Mirror local_dir into the bucket under dest_prefix. Every added path and every
    # deleted path is built from dest_prefix, so this can only ever create, update
    # or delete objects under that one folder - it never touches anything outside it.
    _ensure_bucket()
    dest_prefix = dest_prefix.strip("/")

    local_files = {}
    for root, _dirs, names in os.walk(local_dir):
        for name in names:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, local_dir).replace(os.sep, "/")
            local_files[f"{dest_prefix}/{rel}"] = full

    remote_files = {
        it.path
        for it in api.list_bucket_tree(HF_BUCKET_ID, prefix=dest_prefix, recursive=True)
        if it.type == "file" and it.path.startswith(dest_prefix + "/")
    }

    stale = sorted(remote_files - local_files.keys())
    adds = [(full, remote) for remote, full in sorted(local_files.items())]

    if not adds and not stale:
        return
    api.batch_bucket_files(
        HF_BUCKET_ID,
        add=adds or None,
        delete=stale or None,
    )

async def upload_folder_scoped(local_dir: str, owner: str, repo: str) -> str:
    dest_prefix = f"{owner}/{repo}"
    await asyncio.to_thread(_do_upload_folder_scoped, local_dir, dest_prefix)
    _cache.clear()
    return dest_prefix

async def write_batch_manifest(batch_id: str, files: list[dict]) -> None:
    manifest = {
        "batch_id": batch_id,
        "created": time.time(),
        "files": files,
    }

    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f)
        await asyncio.to_thread(_do_upload, tmp_path, f"_batches/{batch_id}.json")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    _cache.clear()

async def get_batch_manifest(batch_id: str):
    cache_key = f"batch_{batch_id}"
    cached = _get_cache(cache_key)
    if cached is not None:
        return cached

    hf_url = bucket_url(f"_batches/{batch_id}.json")
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(hf_url, headers=auth_headers())
        if r.status_code != 200:
            return None
        try:
            manifest = r.json()
        except ValueError:
            return None

    _set_cache(cache_key, manifest)
    return manifest
