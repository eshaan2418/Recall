"""Repo privacy scan: no images / video / audio / arrays / models / archives / databases anywhere in git history
(or in untracked files that ``git add -A`` would pick up), except an explicit allow-list; no blobs > 2 MB; no
personal-data paths; and no faces in the allow-listed images. Called by scripts/scan_repo.sh (after gitleaks).

  python3 -m recall.privacy.scan_history                # stdlib only: extension + magic-byte + size + path checks
  .venv/bin/python -m recall.privacy.scan_history --faces   # also BlazeFace over every allow-listed raster image

Detection is by extension AND by content (magic bytes), so a JPEG renamed to ``.txt`` is still caught.
Exit status 1 on any finding.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAX_BLOB = 2_000_000

# Paths that may hold binary media. Keep this list short and explicit; every entry must be synthetic or a plot.
ALLOW = re.compile(
    r"^("
    r"tests/fixtures/synthetic/.*"  # generated test fixtures only (never real frames)
    r"|results/[^/]+/figs/.*\.(png|pdf|svg)"  # plots
    r"|paper/figs/.*\.(png|pdf|svg)"
    r"|paper/main\.pdf"
    r"|android/app/src/main/res/(mipmap|drawable)[^/]*/.*\.(png|webp|xml)"  # launcher icons
    r"|android/gradle/wrapper/gradle-wrapper\.jar"
    # synthetic e2e clip rendered by android/tools/make_test_video.py (no camera footage, no people). Exact path only;
    # videos are not face-checked, so every new video needs its own reviewed entry here.
    r"|android/app/src/androidTest/assets/e2e/keys_on_table\.webm"
    r")$"
)
FORBIDDEN_PATH = re.compile(r"^(data/personal/(?!README\.md$)|data/raw/|data/cache/|study/data/)|(^|/)\.env(\..*)?$")
MEDIA_EXT = re.compile(
    r"\.(jpe?g|png|gif|webp|bmp|tiff?|heic|heif|avif|raw|dng"  # images
    r"|mp4|m4v|mov|avi|mkv|webm|mpe?g|3gp|ts|h264|h265|hevc"  # video
    r"|wav|mp3|m4a|aac|flac|ogg|opus"  # audio
    r"|npy|npz|pt|pth|onnx|tflite|safetensors|ckpt|h5|pkl|pickle|bin|litertlm|task"  # arrays / weights
    r"|db|sqlite3?|zip|tar|tgz|gz|bz2|xz|7z|jar|apk|aab"  # containers
    r"|keystore|jks|p12|pfx|pem|key|pdf)$",
    re.I,
)


def magic(head: bytes, size: int) -> str | None:
    """Content type from the first bytes of a blob, or None for text / unknown."""
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] in (b"WEBP", b"AVI ", b"WAVE"):
        return head[8:12].decode().strip().lower()
    if head[4:8] == b"ftyp":
        return "isobmff(mp4/mov/heic/avif)"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "matroska/webm"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:2] == b"BM" and len(head) >= 6 and int.from_bytes(head[2:6], "little") == size:
        return "bmp"
    if head[4:8] == b"TFL3":
        return "tflite"
    if head[:6] == b"\x93NUMPY":
        return "npy"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:16] == b"SQLite format 3\x00":
        return "sqlite"
    if head[:5] == b"%PDF-":
        return "pdf"
    if head[:3] == b"ID3" or head[:4] in (b"OggS", b"fLaC"):
        return "audio"
    if len(head) > 262 and head[257:262] == b"ustar":
        return "tar"
    return None


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True)


def history_blobs() -> dict[str, set[str]]:
    """blob sha -> every path it appeared under, across all refs."""
    out: dict[str, set[str]] = {}
    for c in _git("rev-list", "--all").split():
        for line in _git("ls-tree", "-r", "-z", c).split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            _, kind, sha = meta.split()
            if kind == "blob":
                out.setdefault(sha, set()).add(path)
    return out


def blob_sizes(shas: list[str]) -> dict[str, int]:
    if not shas:
        return {}
    out = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
        cwd=REPO,
        input="\n".join(shas) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {sha: int(n) for sha, n in (ln.split() for ln in out.splitlines())}


def read_blobs(shas: list[str]) -> dict[str, bytes]:
    """Full contents via one ``git cat-file --batch`` process (callers pass only blobs <= MAX_BLOB)."""
    if not shas:
        return {}
    p = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO,
        input="\n".join(shas).encode() + b"\n",
        capture_output=True,
        check=True,
    )
    buf, i, out = p.stdout, 0, {}
    for sha in shas:
        nl = buf.index(b"\n", i)
        size = int(buf[i:nl].split()[2])
        out[sha] = buf[nl + 1 : nl + 1 + size]
        i = nl + 1 + size + 1
    return out


def scan(faces: bool, face_thresh: float = 0.6) -> list[str]:
    problems: list[str] = []
    blobs = history_blobs()
    sizes = blob_sizes(sorted(blobs))
    data = read_blobs(sorted(s for s in blobs if sizes[s] <= MAX_BLOB))
    # (where, path, size, head-or-full-content)
    items: list[tuple[str, str, int, bytes]] = [
        (f"history {sha[:10]}", p, sizes[sha], data.get(sha, b"")) for sha, ps in blobs.items() for p in ps
    ]
    for p in _git("ls-files", "--others", "--exclude-standard", "-z").split("\0"):
        f = REPO / p
        if p and f.is_file():
            n = f.stat().st_size
            items.append(("untracked", p, n, f.read_bytes() if n <= MAX_BLOB else f.open("rb").read(512)))
    allowed_images: list[tuple[str, bytes]] = []
    for where, path, size, content in items:
        kind = magic(content[:512], size)
        allowed = bool(ALLOW.match(path))
        if FORBIDDEN_PATH.search(path):
            problems.append(f"{where}: forbidden path {path}")
        if size > MAX_BLOB:
            problems.append(f"{where}: {path} is {size} bytes (> {MAX_BLOB})")
            continue
        if allowed:
            if kind in ("jpeg", "png", "gif", "webp", "bmp", "tiff"):
                allowed_images.append((f"{where}:{path}", content))
            continue
        if kind:
            problems.append(f"{where}: {path} has {kind} content")
        elif MEDIA_EXT.search(path):
            problems.append(f"{where}: {path} has a media/binary extension")
    if faces and allowed_images:
        problems += face_check(allowed_images, face_thresh)
    print(f"scanned {len(items)} (path, blob) pairs; {len(allowed_images)} allow-listed images")
    return problems


def face_check(images: list[tuple[str, bytes]], thresh: float) -> list[str]:
    import io

    import numpy as np
    from PIL import Image

    from recall.privacy.redact import BlazeFaceDetector

    det = BlazeFaceDetector(thresh=thresh, pad=0.0)
    found = []
    for name, content in images:
        img = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))
        H, W = img.shape[:2]
        # whole image + 2x2 overlapping tiles (BlazeFace squashes its input to 128x128)
        views = [img] + [img[y : y + H * 2 // 3, x : x + W * 2 // 3] for y in (0, H // 3) for x in (0, W // 3)]
        if any(v.shape[0] >= 8 and v.shape[1] >= 8 and det.detect(v) for v in views):
            found.append(f"{name}: face detected (BlazeFace >= {thresh})")
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--faces", action="store_true", help="run BlazeFace over allow-listed images (needs .venv + model)")
    ap.add_argument("--face-thresh", type=float, default=0.6)
    ap.add_argument("--repo", type=Path, default=None, help="repository to scan (default: this one)")
    args = ap.parse_args(argv)
    if args.repo:
        global REPO
        REPO = args.repo.resolve()
    problems = scan(args.faces, args.face_thresh)
    for p in problems:
        print("FAIL", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
