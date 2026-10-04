import os
import re
import time
import uuid
import json
import tempfile
import asyncio
import httpx
from huggingface_hub import HfApi

HF_REPO_ID = os.getenv("HF_REPO_ID", "notamitgamer/cdn")
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_TTL = 60
MAX_CACHE_SIZE = 500
ZIP_MAX_FILES = 300
ZIP_MAX_TOTAL_BYTES = 500 * 1024 * 1024
REPO_STATS_CACHE_TTL = 300
SEARCH_RESULT_LIMIT = 10

api = HfApi(token=HF_TOKEN)

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

        hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{path}"
        async with httpx.AsyncClient(follow_redirects=True) as client:
            r = await client.head(hf_url)
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

def _fetch_tree(path: str):
    return list(api.list_repo_tree(
        repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset", expand=False
    ))

def _fetch_tree_recursive(path: str = ""):
    return list(api.list_repo_tree(
        repo_id=HF_REPO_ID, path_in_repo=path, repo_type="dataset", recursive=True
    ))

_HIDDEN_PREFIXES = ("_batches", "_shortened", "_logs",)

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
            items.append({
                "name": name,
                "path": item.path,
                "is_dir": is_dir,
                "size": size,
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
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo=hf_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

async def upload_temp_file(temp_path: str, filename: str, folder: str = "uploads") -> str:
    slug = str(uuid.uuid4())[:8]
    safe_filename = filename.replace(" ", "_")
    hf_path = f"{folder}/{slug}-{safe_filename}"

    await asyncio.to_thread(_do_upload, temp_path, hf_path)

    _cache.clear()
    return hf_path

def _do_upload_folder_scoped(local_dir: str, dest_prefix: str):
    # path_in_repo + delete_patterns are both scoped to dest_prefix, so this call
    # can only ever create/update/delete files under that one folder - it never
    # touches anything outside of it in the dataset repo.
    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        folder_path=local_dir,
        path_in_repo=dest_prefix,
        delete_patterns="*",
        commit_message=f"gh-sync: update {dest_prefix}",
        token=HF_TOKEN,
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

    hf_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/_batches/{batch_id}.json"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(hf_url)
        if r.status_code != 200:
            return None
        try:
            manifest = r.json()
        except ValueError:
            return None

    _set_cache(cache_key, manifest)
    return manifest
