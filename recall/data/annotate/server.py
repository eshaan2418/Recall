"""Lightweight labelling tool for personal "where is X" ground truth (M9).

    python -m recall.data.annotate.server            # → http://127.0.0.1:8777

Serves the videos in ``data/personal/raw`` *only on localhost* (nothing leaves the machine). While watching,
press **P** (put down) or **T** (picked up) at the moment the hand releases / grabs an object, type the object
and the place ("keys", "hallway shelf"), Enter. Each label is appended to
``data/personal/annotations/<video>.jsonl``. **U** undoes the last label, ←/→ seek ±2 s, space play/pause.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from recall.data.loaders.personal import PERSONAL

RAW, ANN = PERSONAL / "raw", PERSONAL / "annotations"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Recall annotate</title>
<style>body{font:14px system-ui;margin:16px;display:flex;gap:16px}video{width:860px}#side{width:420px}
li{margin:2px 0}input{width:180px}.k{font-family:monospace;background:#eee;padding:0 4px}</style></head><body>
<div><select id="vid"></select><br><video id="v" controls></video>
<p><span class="k">P</span> put down · <span class="k">T</span> picked up · <span class="k">U</span> undo ·
<span class="k">←/→</span> ±2 s · <span class="k">space</span> play/pause</p>
<div id="form" style="display:none">kind <b id="kind"></b> @ <b id="time"></b>s
object <input id="obj" list="objs"> place <input id="place" list="places"> <button id="save">save (Enter)</button></div>
<datalist id="objs"></datalist><datalist id="places"></datalist></div>
<div id="side"><h3>labels (<span id="n">0</span>)</h3><ol id="list"></ol></div>
<script>
const v=document.getElementById('v'),sel=document.getElementById('vid');let labels=[],pending=null;
async function load(){const r=await fetch('/api/labels/'+encodeURIComponent(sel.value));labels=await r.json();
 v.src='/video/'+encodeURIComponent(sel.value);draw();}
function draw(){const ol=document.getElementById('list');ol.innerHTML='';
 labels.forEach(l=>{const li=document.createElement('li');li.textContent=`${l.t.toFixed(1)}s ${l.kind} ${l.object}`+(l.place?` @ ${l.place}`:'');
 li.onclick=()=>{v.currentTime=l.t};ol.appendChild(li)});document.getElementById('n').textContent=labels.length;
 const uniq=k=>[...new Set(labels.map(l=>l[k]).filter(Boolean))];
 document.getElementById('objs').innerHTML=uniq('object').map(o=>`<option value="${o}">`).join('');
 document.getElementById('places').innerHTML=uniq('place').map(o=>`<option value="${o}">`).join('');}
function start(kind){v.pause();pending={video:sel.value,t:+v.currentTime.toFixed(2),kind};
 document.getElementById('kind').textContent=kind;document.getElementById('time').textContent=pending.t;
 document.getElementById('form').style.display='';document.getElementById('obj').focus();}
async function save(){if(!pending)return;pending.object=document.getElementById('obj').value.trim();
 pending.place=document.getElementById('place').value.trim()||null;if(!pending.object)return;
 await fetch('/api/add',{method:'POST',body:JSON.stringify(pending)});labels.push(pending);labels.sort((a,b)=>a.t-b.t);
 pending=null;document.getElementById('form').style.display='none';document.getElementById('obj').value='';draw();v.focus();}
document.getElementById('save').onclick=save;
document.addEventListener('keydown',async e=>{if(e.target.tagName==='INPUT'){if(e.key==='Enter')save();return}
 if(e.key==='p')start('put');else if(e.key==='t')start('take');else if(e.key==='ArrowLeft')v.currentTime-=2;
 else if(e.key==='ArrowRight')v.currentTime+=2;else if(e.key===' '){e.preventDefault();v.paused?v.play():v.pause()}
 else if(e.key==='u'){await fetch('/api/undo/'+encodeURIComponent(sel.value),{method:'POST'});load()}});
fetch('/api/videos').then(r=>r.json()).then(vs=>{sel.innerHTML=vs.map(x=>`<option>${x}</option>`).join('');
 sel.onchange=load;if(vs.length)load();});
</script></body></html>"""


def videos() -> list[str]:
    return sorted(p.name for p in RAW.glob("*") if p.suffix.lower() in VIDEO_EXT) if RAW.exists() else []


def ann_path(video: str) -> Path:
    return ANN / f"{Path(video).stem}.jsonl"


def read_labels(video: str) -> list[dict]:
    p = ann_path(video)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def add_label(d: dict) -> None:
    assert d["kind"] in ("put", "take") and d["video"] in videos() and d["object"]
    ANN.mkdir(parents=True, exist_ok=True)
    rec = {
        "video": Path(d["video"]).stem,
        "t": float(d["t"]),
        "kind": d["kind"],
        "object": str(d["object"])[:80],
        "place": (str(d["place"])[:80] if d.get("place") else None),
    }
    with open(ann_path(d["video"]), "a") as f:
        f.write(json.dumps(rec) + "\n")


def undo(video: str) -> None:
    p = ann_path(video)
    if p.exists():
        lines = [x for x in p.read_text().splitlines() if x.strip()]
        p.write_text("".join(x + "\n" for x in lines[:-1]))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json", extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:  # keep the console quiet
        pass

    def do_GET(self) -> None:
        path = unquote(self.path)
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/videos":
            return self._send(200, json.dumps(videos()).encode())
        if path.startswith("/api/labels/"):
            return self._send(200, json.dumps(read_labels(path[len("/api/labels/") :])).encode())
        if path.startswith("/video/"):
            name = path[len("/video/") :]
            if name not in videos():
                return self._send(404, b"{}")
            return self._video(RAW / name)
        self._send(404, b"{}")

    def _video(self, p: Path) -> None:
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "video/mp4"
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        a, b = 0, size - 1
        if m:
            a = int(m.group(1)) if m.group(1) else 0
            b = min(int(m.group(2)) if m.group(2) else size - 1, a + (8 << 20), size - 1)
        with open(p, "rb") as f:
            f.seek(a)
            body = f.read(b - a + 1)
        extra = {"Accept-Ranges": "bytes"}
        if m:
            extra["Content-Range"] = f"bytes {a}-{b}/{size}"
        self._send(206 if m else 200, body, ctype, extra)

    def do_POST(self) -> None:
        path = unquote(self.path)
        n = int(self.headers.get("Content-Length", 0))
        if path == "/api/add":
            add_label(json.loads(self.rfile.read(n)))
            return self._send(200, b"{}")
        if path.startswith("/api/undo/"):
            undo(path[len("/api/undo/") :])
            return self._send(200, b"{}")
        self._send(404, b"{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    a = ap.parse_args()
    print(f"Recall annotation tool on http://127.0.0.1:{a.port}  ({len(videos())} videos in {RAW})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
