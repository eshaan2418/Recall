"""Compact episodic memory (Contribution 2): SQLite event table + object–place–time graph.

Storage grows with the number of *events*, not hours recorded. Each event row
holds a few scalars, top-k object/location labels, one fp16 embedding per
role (object / location / scene → configurable) and an optional ≤4 KB
redacted thumbnail.

Graph structure (kept as SQL tables so it survives process death on the phone):

* ``objects``  — online clusters of object embeddings (an object "identity")
* ``places``   — online clusters of scene embeddings (user-nameable places)
* ``events``   — (object, place, time) edges with labels and evidence
* ``last_seen``— per object cluster, the most recent ``put`` event (O(1) "where is X")
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recall.memory.schema import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, stream TEXT, t REAL, t_start REAL, kind TEXT, side INTEGER,
  obj_cluster INTEGER, place_id INTEGER, n_obs INTEGER, hand_score REAL, release_obs INTEGER,
  obj_top TEXT, loc_top TEXT, box TEXT, obj_emb BLOB, loc_emb BLOB, scene_emb BLOB, thumb BLOB,
  deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ev_t ON events(stream, t);
CREATE INDEX IF NOT EXISTS ev_obj ON events(obj_cluster, t);
CREATE TABLE IF NOT EXISTS objects (id INTEGER PRIMARY KEY, centroid BLOB, n INTEGER, label TEXT);
CREATE TABLE IF NOT EXISTS places (id INTEGER PRIMARY KEY, centroid BLOB, n INTEGER, name TEXT);
CREATE TABLE IF NOT EXISTS last_seen (obj_cluster INTEGER PRIMARY KEY, event_id INTEGER, t REAL);
CREATE TABLE IF NOT EXISTS settings (k TEXT PRIMARY KEY, v TEXT);
"""


@dataclass
class StoreConfig:
    store_obj_emb: bool = True
    store_loc_emb: bool = True
    store_scene_emb: bool = False  # scene info is kept via place_id
    emb_dtype: str = "float16"  # float16 | int8
    store_thumbs: bool = True
    obj_cluster_sim: float = 0.80
    place_cluster_sim: float = 0.80
    merge_window_s: float = 10.0
    merge_sim: float = 0.85
    retention_days: float | None = None  # auto-delete older events (except each object's last put)


def _enc(x: np.ndarray | None, dtype: str) -> bytes | None:
    if x is None:
        return None
    x = np.asarray(x, np.float32)
    if dtype == "int8":
        s = float(np.abs(x).max() or 1.0)
        return np.float32(s).tobytes() + np.round(x / s * 127).astype(np.int8).tobytes()
    return x.astype(np.float16).tobytes()


def _dec(b: bytes | None, dtype: str) -> np.ndarray | None:
    if b is None:
        return None
    if dtype == "int8":
        s = np.frombuffer(b[:4], np.float32)[0]
        return np.frombuffer(b[4:], np.int8).astype(np.float32) * s / 127
    return np.frombuffer(b, np.float16).astype(np.float32)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:", cfg: StoreConfig | None = None):
        self.cfg = cfg or StoreConfig()
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self._obj_c: list[np.ndarray] = []
        self._obj_n: list[int] = []
        self._place_c: list[np.ndarray] = []
        self._place_n: list[int] = []

    # ---------------- clustering ----------------
    @staticmethod
    def _assign(cents: list[np.ndarray], counts: list[int], x: np.ndarray, thr: float) -> int:
        if cents:
            C = np.stack(cents)
            s = C @ x
            j = int(np.argmax(s))
            if s[j] >= thr:
                counts[j] += 1
                c = cents[j] + (x - cents[j]) / counts[j]
                cents[j] = c / np.linalg.norm(c)
                return j
        cents.append(x / np.linalg.norm(x))
        counts.append(1)
        return len(cents) - 1

    # ---------------- writes ----------------
    def add(self, e: Event) -> int:
        cfg = self.cfg
        oc = self._assign(self._obj_c, self._obj_n, e.obj_emb, cfg.obj_cluster_sim)
        pc = self._assign(self._place_c, self._place_n, e.scene_emb, cfg.place_cluster_sim)
        e.place_id = pc
        # merge: same object cluster, same kind, within window → replace the older row
        row = self.db.execute(
            "SELECT id, obj_emb FROM events WHERE stream=? AND kind=? AND obj_cluster=? AND t>=? AND deleted=0 "
            "ORDER BY t DESC LIMIT 1",
            (e.stream_id, e.kind, oc, e.t - cfg.merge_window_s),
        ).fetchone()
        if row is not None and e.kind == "put":
            prev = _dec(row[1], cfg.emb_dtype)
            if prev is not None and float(prev @ e.obj_emb) / (np.linalg.norm(prev) + 1e-8) >= cfg.merge_sim:
                self.db.execute("UPDATE events SET deleted=1 WHERE id=?", (row[0],))
        cur = self.db.execute(
            "INSERT INTO events (stream,t,t_start,kind,side,obj_cluster,place_id,n_obs,hand_score,release_obs,"
            "obj_top,loc_top,box,obj_emb,loc_emb,scene_emb,thumb) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                e.stream_id,
                e.t,
                e.t_start,
                e.kind,
                e.side,
                oc,
                pc,
                e.n_obs,
                e.hand_score,
                int(e.release_observed),
                json.dumps([[int(a), round(float(b), 4)] for a, b in e.obj_top]),
                json.dumps([[int(a), round(float(b), 4)] for a, b in e.loc_top]),
                json.dumps([round(float(x), 4) for x in e.box]),
                _enc(e.obj_emb, cfg.emb_dtype) if cfg.store_obj_emb else None,
                _enc(e.loc_emb, cfg.emb_dtype) if cfg.store_loc_emb else None,
                _enc(e.scene_emb, cfg.emb_dtype) if cfg.store_scene_emb else None,
                e.thumb if cfg.store_thumbs else None,
            ),
        )
        eid = int(cur.lastrowid or 0)
        e.event_id = eid
        if e.kind == "put":
            self.db.execute("INSERT OR REPLACE INTO last_seen VALUES (?,?,?)", (oc, eid, e.t))
        return eid

    def add_all(self, events: Iterable[Event]) -> None:
        for e in events:
            self.add(e)
        self.commit()

    def commit(self) -> None:
        # persist cluster centroids
        self.db.execute("DELETE FROM objects")
        self.db.executemany(
            "INSERT INTO objects VALUES (?,?,?,NULL)",
            [(i, _enc(c, "float16"), n) for i, (c, n) in enumerate(zip(self._obj_c, self._obj_n))],
        )
        self.db.execute("DELETE FROM places")
        self.db.executemany(
            "INSERT INTO places VALUES (?,?,?,NULL)",
            [(i, _enc(c, "float16"), n) for i, (c, n) in enumerate(zip(self._place_c, self._place_n))],
        )
        self.db.commit()

    # ---------------- retention / forgetting ----------------
    def apply_retention(self, now_t: float) -> int:
        if self.cfg.retention_days is None:
            return 0
        cutoff = now_t - self.cfg.retention_days * 86400
        keep = {r[0] for r in self.db.execute("SELECT event_id FROM last_seen")}
        ids = [
            r[0]
            for r in self.db.execute("SELECT id FROM events WHERE t<? AND deleted=0", (cutoff,))
            if r[0] not in keep
        ]
        self.db.executemany(
            "UPDATE events SET deleted=1, obj_emb=NULL, loc_emb=NULL, scene_emb=NULL, thumb=NULL WHERE id=?",
            [(i,) for i in ids],
        )
        self.db.commit()
        return len(ids)

    def forget(self, event_id: int) -> None:
        """User-initiated 'forget this': hard delete the row and its evidence."""
        self.db.execute("DELETE FROM events WHERE id=?", (event_id,))
        self.db.execute("DELETE FROM last_seen WHERE event_id=?", (event_id,))
        self.db.commit()

    def vacuum(self) -> None:
        self.db.execute("DELETE FROM events WHERE deleted=1")
        self.db.commit()
        self.db.execute("VACUUM")

    # ---------------- reads ----------------
    def events(
        self, before_t: float | None = None, kinds: tuple[str, ...] | None = None, stream: str | None = None
    ) -> list[dict]:
        q = "SELECT * FROM events WHERE deleted=0"
        args: list = []
        if stream is not None:
            q += " AND stream=?"
            args.append(stream)
        if before_t is not None:
            q += " AND t<?"
            args.append(before_t)
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        q += " ORDER BY t"
        cols = [c[0] for c in self.db.execute("SELECT * FROM events LIMIT 0").description]
        out = []
        for r in self.db.execute(q, args):
            d = dict(zip(cols, r))
            d["obj_top"] = json.loads(d["obj_top"])
            d["loc_top"] = json.loads(d["loc_top"])
            d["box"] = json.loads(d["box"])
            d["obj_emb"] = _dec(d["obj_emb"], self.cfg.emb_dtype)
            d["loc_emb"] = _dec(d["loc_emb"], self.cfg.emb_dtype)
            d["scene_emb"] = _dec(d["scene_emb"], self.cfg.emb_dtype)
            out.append(d)
        return out

    def n_events(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM events WHERE deleted=0").fetchone()[0])

    def size_bytes(self) -> int:
        """Bytes on disk after VACUUM (for ':memory:' stores, serialise to measure)."""
        self.commit()
        if self.path == ":memory:":
            # payload bytes + a fixed per-row overhead estimate (SQLite row header, scalars)
            tot = 0
            for (n,) in self.db.execute(
                "SELECT COALESCE(LENGTH(obj_emb),0)+COALESCE(LENGTH(loc_emb),0)+COALESCE(LENGTH(scene_emb),0)+"
                "COALESCE(LENGTH(thumb),0)+LENGTH(obj_top)+LENGTH(loc_top)+LENGTH(box)+64 FROM events WHERE deleted=0"
            ):
                tot += int(n)
            tot += sum(len(c) * 2 + 16 for c in self._obj_c) + sum(len(c) * 2 + 16 for c in self._place_c)
            return tot
        self.vacuum()
        return Path(self.path).stat().st_size


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - t0) * 1e3
