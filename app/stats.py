"""Public stats page: what is in the CDN (types, sizes, folders, growth).

Everything is computed from the file list storage.py already caches, then kept
for STATS_TTL seconds, so viewing the page never adds load on the bucket.
Only aggregate numbers are exposed - no IPs, no upload log.
"""
from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from . import storage

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

STATS_TTL = 600
_cache: dict = {"at": 0.0, "data": None}
_lock = asyncio.Lock()

CATEGORIES = [
    ("images", "Images", "png jpg jpeg gif webp avif svg ico bmp heic tiff"),
    ("documents", "Documents", "pdf doc docx ppt pptx xls xlsx odt ods odp txt md rtf csv epub"),
    ("code", "Code & data", "js mjs css html htm json xml yml yaml toml py ts tsx jsx c cpp h java go rs sh sql map wasm"),
    ("audio", "Audio", "mp3 wav ogg flac m4a aac opus"),
    ("video", "Video", "mp4 mkv webm mov avi m4v"),
    ("archives", "Archives", "zip tar gz tgz 7z rar xz bz2"),
    ("fonts", "Fonts", "woff woff2 ttf otf eot"),
]
_EXT_TO_CAT = {e: (k, label) for k, label, exts in CATEGORIES for e in exts.split()}
SIZE_BUCKETS = [(10 * 1024, "< 10 KB"), (100 * 1024, "10-100 KB"), (1024 ** 2, "100 KB-1 MB"),
                (10 * 1024 ** 2, "1-10 MB"), (100 * 1024 ** 2, "10-100 MB"), (float("inf"), "> 100 MB")]
TOP_LEVEL_FOLDERS = {"uploads", "third-party"}


def _ext(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name.strip(".") else ""


def _folder(path: str) -> str:
    parts = path.split("/")
    if len(parts) == 1:
        return "(root)"
    return parts[0] if parts[0] in TOP_LEVEL_FOLDERS else "/".join(parts[:2]) if len(parts) > 2 else parts[0]


def compute(items: list[dict]) -> dict:
    files = [i for i in items if not i["is_dir"] and i["name"] != ".gitattributes"]
    if not files:
        return {"empty": True}

    sizes = [f["size"] for f in files]
    cats: dict[str, dict] = {}
    exts: dict[str, list] = defaultdict(lambda: [0, 0])
    folders: dict[str, list] = defaultdict(lambda: [0, 0])
    buckets = Counter()
    months: dict[str, list] = defaultdict(lambda: [0, 0])

    for f in files:
        ext = _ext(f["name"])
        key, label = _EXT_TO_CAT.get(ext, ("other", "Other"))
        c = cats.setdefault(key, {"key": key, "label": label, "files": 0, "bytes": 0})
        c["files"] += 1
        c["bytes"] += f["size"]
        exts[ext or "(none)"][0] += 1
        exts[ext or "(none)"][1] += f["size"]
        fo = folders[_folder(f["path"])]
        fo[0] += 1
        fo[1] += f["size"]
        buckets[next(lbl for limit, lbl in SIZE_BUCKETS if f["size"] < limit)] += 1
        if f.get("ts"):
            m = months[datetime.fromtimestamp(f["ts"], timezone.utc).strftime("%Y-%m")]
            m[0] += 1
            m[1] += f["size"]

    top_ext = sorted(exts.items(), key=lambda kv: kv[1][0], reverse=True)
    top_folders = sorted(folders.items(), key=lambda kv: kv[1][1], reverse=True)
    largest = sorted(files, key=lambda f: f["size"], reverse=True)[:8]
    month_rows = [{"m": k, "files": v[0], "bytes": v[1]} for k, v in sorted(months.items())][-12:]

    return {
        "empty": False,
        "generated": int(time.time()),
        "totals": {
            "files": len(files), "bytes": sum(sizes), "avg": int(sum(sizes) / len(sizes)),
            "median": int(statistics.median(sizes)), "types": len(exts),
        },
        "categories": sorted(cats.values(), key=lambda c: c["files"], reverse=True),
        "extensions": [{"ext": e, "files": v[0], "bytes": v[1]} for e, v in top_ext[:10]],
        "extensions_more": sum(v[0] for _, v in top_ext[10:]),
        "sizes": [{"label": lbl, "files": buckets.get(lbl, 0)} for _, lbl in SIZE_BUCKETS],
        "folders": [{"name": n, "files": v[0], "bytes": v[1]} for n, v in top_folders[:8]],
        "largest": [{"name": f["name"], "path": f["path"], "size": f["size"]} for f in largest],
        "months": month_rows,
    }


async def get_stats() -> dict:
    now = time.time()
    if _cache["data"] is not None and now - _cache["at"] < STATS_TTL:
        return _cache["data"]
    async with _lock:
        if _cache["data"] is not None and time.time() - _cache["at"] < STATS_TTL:
            return _cache["data"]
        items = await storage._get_full_tree()
        if not items:  # listing failed: keep serving the last good result if we have one
            return _cache["data"] or {"empty": True}
        data = await asyncio.to_thread(compute, items)
        _cache.update(at=time.time(), data=data)
        return data


@router.get("/stats")
async def stats_page(request: Request):
    return templates.TemplateResponse(request, "stats.html", {"page": "stats"})


@router.get("/api/stats")
async def stats_api():
    return JSONResponse(await get_stats(), headers={"Cache-Control": "public, max-age=300"})
