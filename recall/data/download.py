"""Resumable, checksum-verified, parallel-range HTTP downloader.

Used for EPIC-KITCHENS-100 (publicly downloadable from data.bris.ac.uk under
CC BY-NC 4.0). Files are downloaded in fixed-size chunks with several
concurrent range requests; completed chunks are tracked in a sidecar
``.parts`` directory so an interrupted download resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CHUNK = 16 * 1024 * 1024


def remote_size(url: str, session: requests.Session) -> int:
    r = session.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def _fetch_chunk(url: str, start: int, end: int, path: Path, retries: int = 8) -> None:
    for attempt in range(retries):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = path.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for buf in r.iter_content(1 << 20):
                        f.write(buf)
                if tmp.stat().st_size != end - start + 1:
                    raise OSError(f"short chunk {tmp.stat().st_size} != {end - start + 1}")
                tmp.rename(path)
                return
        except (requests.RequestException, OSError) as e:  # pragma: no cover - network
            wait = min(60, 2**attempt)
            print(f"  retry {attempt + 1} {path.name}: {e}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed chunk {path}")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 24), b""):
            h.update(buf)
    return h.hexdigest()


def download(url: str, dest: Path, md5: str | None = None, workers: int = 8) -> Path:
    """Download ``url`` to ``dest`` (resumable). Verifies md5 if given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done_marker = dest.with_name(dest.name + ".ok")
    if dest.exists() and done_marker.exists():
        return dest
    session = requests.Session()
    size = remote_size(url, session)
    parts = dest.with_name(dest.name + ".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    todo = [
        (i, s, e)
        for i, (s, e) in enumerate(ranges)
        if not (parts / f"{i:06d}").exists() or (parts / f"{i:06d}").stat().st_size != e - s + 1
    ]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch_chunk, url, s, e, parts / f"{i:06d}") for i, s, e in todo]
        for n, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if n % 20 == 0 or n == len(futs):
                mb = n * CHUNK / 1e6
                print(f"  {dest.name}: {n}/{len(futs)} chunks, {mb / (time.time() - t0):.1f} MB/s")
    tmp = dest.with_name(dest.name + ".assembling")
    with open(tmp, "wb") as out:
        for i in range(len(ranges)):
            with open(parts / f"{i:06d}", "rb") as f:
                shutil.copyfileobj(f, out)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {dest}")
    if md5 is not None:
        got = md5sum(tmp)
        if got != md5:
            shutil.rmtree(parts)
            tmp.unlink()
            raise RuntimeError(f"md5 mismatch for {dest}: {got} != {md5}")
    tmp.rename(dest)
    shutil.rmtree(parts)
    done_marker.write_text(json.dumps({"url": url, "size": size, "md5": md5}))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest", type=Path)
    ap.add_argument("--md5")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RECALL_DL_WORKERS", 8)))
    a = ap.parse_args()
    download(a.url, a.dest, a.md5, a.workers)


if __name__ == "__main__":
    main()
