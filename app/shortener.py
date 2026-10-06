import random
import string
import asyncio
import re
import httpx

from .storage import HF_TOKEN, bucket_url, auth_headers, _do_upload_bytes

# Reusable HTTP client for fast connection pooling & HTTP keep-alive
_client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)

# Alphanumeric validator to guard against path traversal attempts
_ID_REGEX = re.compile(r"^[a-z0-9]{4,16}$")

def generate_id(length: int = 8) -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))

async def shorten_url(destination_url: str) -> str:
    """Save a destination URL as a tiny object in the Hugging Face bucket."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not configured.")

    destination_url = destination_url.strip()

    # Collision-check loop with a safeguard cutoff
    attempts = 0
    while True:
        short_id = generate_id()
        r = await _client.head(bucket_url(f"_shortened/{short_id}"), headers=auth_headers())
        if r.status_code == 404:
            break
        attempts += 1
        if attempts > 5:
            # Fallback to longer ID if collisions occur
            short_id = generate_id(length=12)
            break

    # Upload straight from memory without writing to disk
    await asyncio.to_thread(
        _do_upload_bytes,
        destination_url.encode("utf-8"),
        f"_shortened/{short_id}",
    )
    return short_id

async def get_destination_url(short_id: str) -> str | None:
    """Look up the destination URL directly from the bucket's resolve endpoint."""
    short_id = short_id.strip().lower()
    if not _ID_REGEX.match(short_id):
        return None

    try:
        r = await _client.get(bucket_url(f"_shortened/{short_id}"), headers=auth_headers())
        if r.status_code == 200:
            return r.text.strip()
    except httpx.RequestError:
        return None
    return None

def list_all_ids() -> list[str]:
    """Stub kept for router compatibility."""
    return []
