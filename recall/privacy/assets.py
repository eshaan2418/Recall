"""Checksum-pinned downloads for the privacy layer (M7).

Nothing fetched here is committed: ``models/`` and ``data/raw/`` are git-ignored.

  PYTHONPATH=. .venv/bin/python -m recall.privacy.assets face_detector      # BlazeFace short-range (Apache-2.0)
  PYTHONPATH=. .venv/bin/python -m recall.privacy.assets wider_val          # WIDER FACE val (research, non-commercial)

Each file is verified against the sha256 recorded below (for the Hugging Face mirror of WIDER FACE this is
the git-LFS object id, i.e. the sha256 of the file). A mismatch deletes the file and fails.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from recall.paths import DATA, MODELS


@dataclass(frozen=True)
class Asset:
    url: str
    sha256: str
    dest: Path
    licence: str
    unzip: bool = False


FACE_MODEL = MODELS / "litert" / "face_detector.tflite"
WIDER_DIR = DATA / "raw" / "wider_face"
_HF = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"

ASSETS: dict[str, list[Asset]] = {
    # Official MediaPipe model storage, pinned to version 1 (not "latest"). Same file the Android app ships
    # as assets/face_detector.tflite. Model card: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
    "face_detector": [
        Asset(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/"
            "blaze_face_short_range.tflite",
            "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
            FACE_MODEL,
            "Apache-2.0 (Google MediaPipe)",
        )
    ],
    # WIDER FACE (Yang et al., CVPR 2016), validation split + annotations. Licence on the HF mirror:
    # CC BY-NC-ND 4.0; images are from WIDER and are for non-commercial research use. Evaluation only;
    # never redistributed, never committed.
    "wider_val": [
        Asset(
            f"{_HF}/wider_face_split.zip",
            "c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c",
            WIDER_DIR / "wider_face_split.zip",
            "CC BY-NC-ND 4.0 (WIDER FACE)",
            unzip=True,
        ),
        Asset(
            f"{_HF}/WIDER_val.zip",
            "f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a",
            WIDER_DIR / "WIDER_val.zip",
            "CC BY-NC-ND 4.0 (WIDER FACE)",
            unzip=True,
        ),
    ],
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            h.update(buf)
    return h.hexdigest()


def fetch(asset: Asset) -> Path:
    import requests

    if not (asset.dest.exists() and sha256sum(asset.dest) == asset.sha256):
        asset.dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = asset.dest.with_suffix(asset.dest.suffix + ".part")
        with requests.get(asset.url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for buf in r.iter_content(1 << 20):
                    f.write(buf)
        got = sha256sum(tmp)
        if got != asset.sha256:
            tmp.unlink()
            raise RuntimeError(f"sha256 mismatch for {asset.url}: {got} != {asset.sha256}")
        tmp.rename(asset.dest)
    if asset.unzip:
        with zipfile.ZipFile(asset.dest) as z:
            root = asset.dest.parent.resolve()
            for m in z.namelist():  # refuse path traversal in archives
                if not (root / m).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in {asset.dest.name}: {m}")
            z.extractall(root)
    print(f"ok {asset.dest} sha256={asset.sha256} licence={asset.licence}")
    return asset.dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(ASSETS))
    args = ap.parse_args(argv)
    for n in args.names:
        for a in ASSETS[n]:
            fetch(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
