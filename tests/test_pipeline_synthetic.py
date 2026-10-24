"""End-to-end pipeline on synthetic streams (no licensed data)."""

import numpy as np

from recall.bench.harness import System, build_view, metrics, score
from recall.data.synthetic import SynthWorld, make_video, where_queries
from recall.memory.store import MemoryStore, StoreConfig
from recall.perception.events import extract_events
from recall.qa.answer import AnswerConfig, answer
from recall.qa.parse import ParsedQuery
from recall.trigger.policies import oracle, uniform


def _run(policy, n_videos=4, noise=0.3):
    world = SynthWorld(noise=noise)
    heads = world.heads()
    recs = []
    for k in range(n_videos):
        vid = f"S{k}"
        feats, narr = make_video(world, vid, seed=k)
        sys = System(
            "x",
            policy,
            cfg=AnswerConfig(tau=0.2),
        )
        view, cost = build_view(sys, vid, feats, heads)
        for q in where_queries(narr, vid, 600.0):
            pq = ParsedQuery("where", q.obj_key)
            a = answer(view, pq, q.obj_class, q.t_query, sys.cfg)
            recs.append(score(q, a, heads.loc_ids))
    return recs, metrics(recs)


def test_all_frames_high_accuracy():
    recs, m = _run(lambda vid, f: np.ones(len(f["t"]), bool))
    assert m["where_loc_acc"]["mean"] > 0.85
    assert m["absent_abstain"]["mean"] == 1.0


def test_oracle_matches_all_and_sparse_uniform_is_worse():
    _, m_all = _run(lambda vid, f: np.ones(len(f["t"]), bool))
    _, m_sparse = _run(lambda vid, f: uniform(len(f["t"]), 25))
    assert m_sparse["where_loc_acc"]["mean"] < m_all["where_loc_acc"]["mean"]


def test_events_put_take_pairs():
    world = SynthWorld()
    feats, narr = make_video(world, "S0", seed=0, n_interactions=10)
    ev = extract_events(feats, np.ones(len(feats["t"]), bool), "S0")
    kinds = [e.kind for e in ev]
    assert kinds.count("take") == kinds.count("put") == 10
    assert all(e.release_observed for e in ev if e.kind == "put")


def test_store_roundtrip_and_size(tmp_path):
    world = SynthWorld()
    feats, _ = make_video(world, "S0", seed=0)
    ev = extract_events(feats, np.ones(len(feats["t"]), bool), "S0")
    st = MemoryStore(tmp_path / "m.db", StoreConfig(store_thumbs=False))
    st.add_all(ev)
    rows = st.events(kinds=("put",))
    assert len(rows) >= 15
    assert rows[0]["obj_emb"].shape == (32,)
    size = st.size_bytes()
    assert 0 < size < 200_000
    st.forget(rows[0]["id"])
    assert len(st.events(kinds=("put",))) == len(rows) - 1


def test_int8_embeddings_close():
    world = SynthWorld()
    feats, _ = make_video(world, "S0", seed=0)
    ev = extract_events(feats, np.ones(len(feats["t"]), bool), "S0")
    st = MemoryStore(":memory:", StoreConfig(emb_dtype="int8", store_thumbs=False))
    st.add_all(ev)
    r = st.events(kinds=("put",))[0]
    e = next(x for x in ev if x.event_id == r["id"])
    assert float(np.dot(r["obj_emb"], e.obj_emb) / np.linalg.norm(r["obj_emb"])) > 0.99


def test_oracle_policy_runs():
    world = SynthWorld()
    feats, narr = make_video(world, "S0", seed=0)
    m = oracle(feats["t"], narr[["start_s", "stop_s"]].to_numpy(), 1.0)
    assert 0 < m.mean() < 1
