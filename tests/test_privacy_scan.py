"""Repo privacy scan (recall/privacy/scan_history.py, scripts/gitleaks.toml) on throwaway git repos."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recall.privacy import scan_history

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
REPO = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "README.md").write_text("hello\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "init")
    return r


def _png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), 200, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _scan(r: Path) -> int:
    return scan_history.main(["--repo", str(r)])


def test_clean_repo_passes(tmp_path):
    assert _scan(_repo(tmp_path)) == 0


def test_image_deleted_from_history_still_fails(tmp_path, capsys):
    r = _repo(tmp_path)
    (r / "notes.txt").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 64)  # JPEG disguised as text
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "oops")
    _git(r, "rm", "-q", "notes.txt")
    _git(r, "commit", "-qm", "remove")
    assert _scan(r) == 1
    assert "jpeg content" in capsys.readouterr().out


def test_extension_and_paths_and_untracked(tmp_path, capsys):
    r = _repo(tmp_path)
    (r / "clip.mp4").write_bytes(b"not really a video")
    assert _scan(r) == 1, "untracked media would be picked up by git add -A"
    (r / "clip.mp4").unlink()
    (r / "data" / "personal").mkdir(parents=True)
    (r / "data" / "personal" / "log.csv").write_text("t,x\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "personal")
    assert _scan(r) == 1
    assert "forbidden path data/personal/log.csv" in capsys.readouterr().out


def test_allow_listed_synthetic_fixture_passes(tmp_path):
    r = _repo(tmp_path)
    d = r / "tests" / "fixtures" / "synthetic"
    d.mkdir(parents=True)
    (d / "flat.png").write_bytes(_png())
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "fixture")
    assert _scan(r) == 0
    (r / "elsewhere.png").write_bytes(_png())
    assert _scan(r) == 1


def test_magic_detection():
    assert scan_history.magic(b"\x89PNG\r\n\x1a\n....", 100) == "png"
    assert scan_history.magic(b"\0\0\0\x18ftypmp42", 100).startswith("isobmff")
    assert scan_history.magic(b"RIFF\0\0\0\0WEBPVP8 ", 100) == "webp"
    assert scan_history.magic(b"SQLite format 3\x00", 100) == "sqlite"
    assert scan_history.magic(b"BM\x10\x00\x00\x00", 16) == "bmp"
    assert scan_history.magic(b"BMW is a car", 12) is None
    assert scan_history.magic(b"plain text", 10) is None


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed")
def test_gitleaks_config_allows_provenance_sha_only(tmp_path):
    cfg = REPO / "scripts" / "gitleaks.toml"
    sha = "0276c24a20ab4013d994eeafe3b7d5bdb57809b0"
    (tmp_path / "ok.json").write_text(f'{{\n "evidence_stage": "x",\n "git_commit": "{sha}",\n "seed": 0\n}}\n')
    ok = subprocess.run(["gitleaks", "dir", str(tmp_path), "--config", str(cfg), "--no-banner"], capture_output=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    (tmp_path / "bad.py").write_text('AWS_SECRET_ACCESS_KEY = "' + "wJalrXUtnFEMI/K7MDENG/bPxRfiCYzEXAMPLEKEY" + '"\n')
    (tmp_path / "bad2.env").write_text("GITHUB_TOKEN=ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8" + "\n")
    bad = subprocess.run(["gitleaks", "dir", str(tmp_path), "--config", str(cfg), "--no-banner"], capture_output=True)
    assert bad.returncode != 0
