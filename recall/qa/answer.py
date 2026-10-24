"""Query answering over episodic memory: retrieval, structured answers, confidence features.

The answerer only uses what the memory stores (events with embeddings and
top-k labels), never the video. Every answer carries the supporting event so
the UI can show the evidence thumbnail and timestamp, and a feature vector for
the calibrator (:mod:`recall.qa.calibrate`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recall.memory.schema import Event
from recall.perception.heads import BlendHead
from recall.qa.parse import ParsedQuery

KIND_FOR_VERB = {
    "take": ("take",),
    "put": ("put",),
    "open": ("touch", "take"),
    "close": ("touch", "take"),
    "turn-on": ("touch",),
    "turn-off": ("touch",),
    "wash": ("take", "touch"),
}

FEATURES = [
    "score",
    "is_argmax",
    "score_margin",
    "loc_p",
    "loc_margin",
    "log_n_obs",
    "log_age_s",
    "n_conflicts",
    "second_score",
    "release_obs",
    "hand_score",
    "n_candidates",
]


@dataclass
class MemoryView:
    """Column view of the events of one stream, with per-event class log-probs computed from stored embeddings."""

    t: np.ndarray
    kind: np.ndarray
    n_obs: np.ndarray
    hand_score: np.ndarray
    release_obs: np.ndarray
    obj_logp: np.ndarray  # [N, n_nouns]
    loc_logp: np.ndarray  # [N, n_locs]
    obj_emb: np.ndarray
    loc_emb: np.ndarray
    scene_emb: np.ndarray
    events: list[Event] = field(default_factory=list)

    @staticmethod
    def build(events: list[Event], obj_head: BlendHead, loc_head: BlendHead) -> MemoryView:
        n = len(events)
        D = obj_head.text_bank.shape[1]
        if n == 0:
            z = np.zeros((0, D), np.float32)
            return MemoryView(
                np.zeros(0),
                np.zeros(0, "<U5"),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros((0, obj_head.text_bank.shape[0])),
                np.zeros((0, loc_head.text_bank.shape[0])),
                z,
                z,
                z,
                [],
            )
        O = np.stack([e.obj_emb for e in events]).astype(np.float32)
        L = np.stack([e.loc_emb for e in events]).astype(np.float32)
        S = np.stack([e.scene_emb for e in events]).astype(np.float32)
        return MemoryView(
            t=np.array([e.t for e in events]),
            kind=np.array([e.kind for e in events]),
            n_obs=np.array([e.n_obs for e in events], float),
            hand_score=np.array([e.hand_score for e in events], float),
            release_obs=np.array([e.release_observed for e in events], float),
            obj_logp=obj_head.log_probs(O),
            loc_logp=loc_head.log_probs(L),
            obj_emb=O,
            loc_emb=L,
            scene_emb=S,
            events=events,
        )


@dataclass
class AnswerConfig:
    mode: str = "latest"  # latest | recency
    tau: float = 0.05  # min p(object | event) to be a candidate
    recency_s: float = 600.0  # for mode=recency: score * exp(-age/recency_s)
    use_kinds_where: tuple[str, ...] = ("put",)
    fallback_take: bool = True  # if no put candidate, fall back to last take/touch ("you had it in hand at …")
    kinds_override: tuple[str, ...] | None = None  # B2: every stored item is a "frame"


@dataclass
class Answer:
    kind: str
    abstain: bool
    event_idx: int | None = None
    t: float | None = None
    loc_idx: int | None = None
    state: str | None = None
    feats: dict = field(default_factory=dict)
    confidence: float | None = None


def _pick(scores: np.ndarray, t: np.ndarray, t_q: float, cfg: AnswerConfig) -> tuple[int | None, np.ndarray]:
    cand = np.flatnonzero(scores >= cfg.tau)
    if len(cand) == 0:
        return None, cand
    if cfg.mode == "latest":
        return int(cand[np.argmax(t[cand])]), cand
    w = scores[cand] * np.exp(-(t_q - t[cand]) / cfg.recency_s)
    return int(cand[np.argmax(w)]), cand


def _features(view: MemoryView, j: int, cls: int, scores: np.ndarray, cand: np.ndarray, t_q: float, tau: float) -> dict:
    lp = view.obj_logp[j]
    s = float(scores[j])
    others = np.delete(np.exp(lp), cls)
    locp = np.exp(view.loc_logp[j])
    srt = np.sort(locp)[::-1]
    later = cand[(view.t[cand] > view.t[j] - 1e-6) & (cand != j)]
    rest = np.delete(scores[cand], np.flatnonzero(cand == j)) if len(cand) else np.zeros(0)
    return {
        "score": s,
        "is_argmax": float(int(np.argmax(lp)) == cls),
        "score_margin": s - float(others.max(initial=0.0)),
        "loc_p": float(srt[0]),
        "loc_margin": float(srt[0] - (srt[1] if len(srt) > 1 else 0.0)),
        "log_n_obs": float(np.log1p(view.n_obs[j])),
        "log_age_s": float(np.log1p(max(0.0, t_q - view.t[j]))),
        "n_conflicts": float(len(later) + (np.sum(rest >= tau) if len(rest) else 0)),
        "second_score": float(rest.max(initial=0.0)),
        "release_obs": float(view.release_obs[j]),
        "hand_score": float(view.hand_score[j]),
        "n_candidates": float(len(cand)),
    }


def answer_where(view: MemoryView, cls: int, t_q: float, cfg: AnswerConfig) -> Answer:
    before = view.t < t_q
    kinds = np.isin(view.kind, cfg.kinds_override or cfg.use_kinds_where) & before
    scores = np.where(kinds, np.exp(view.obj_logp[:, cls]) if len(view.t) else 0.0, 0.0)
    j, cand = _pick(scores, view.t, t_q, cfg)
    if j is None and cfg.fallback_take and cfg.kinds_override is None:
        kinds = np.isin(view.kind, ("take", "touch")) & before
        scores = np.where(kinds, np.exp(view.obj_logp[:, cls]) if len(view.t) else 0.0, 0.0)
        j, cand = _pick(scores, view.t, t_q, cfg)
    if j is None:
        return Answer("where", True, feats={k: 0.0 for k in FEATURES})
    f = _features(view, j, cls, scores, cand, t_q, cfg.tau)
    f["fallback"] = float(view.kind[j] != "put")
    return Answer("where", False, j, float(view.t[j]), int(np.argmax(view.loc_logp[j])), feats=f)


def answer_temporal(view: MemoryView, cls: int, verb: str | None, t_q: float, cfg: AnswerConfig) -> Answer:
    kinds = cfg.kinds_override or KIND_FOR_VERB.get(verb or "take", ("take",))
    m = np.isin(view.kind, kinds) & (view.t < t_q)
    scores = np.where(m, np.exp(view.obj_logp[:, cls]) if len(view.t) else 0.0, 0.0)
    j, cand = _pick(scores, view.t, t_q, cfg)
    if j is None:
        return Answer("temporal", True, feats={k: 0.0 for k in FEATURES})
    f = _features(view, j, cls, scores, cand, t_q, cfg.tau)
    return Answer("temporal", False, j, float(view.t[j]), feats=f)


def answer_state(view: MemoryView, cls: int, t_q: float, cfg: AnswerConfig, state_head=None) -> Answer:
    """Last observed interaction with an appliance + predicted on/off after it."""
    m = np.isin(view.kind, cfg.kinds_override or ("touch", "take")) & (view.t < t_q)
    scores = np.where(m, np.exp(view.obj_logp[:, cls]) if len(view.t) else 0.0, 0.0)
    j, cand = _pick(scores, view.t, t_q, cfg)
    if j is None:
        return Answer("state", True, feats={k: 0.0 for k in FEATURES})
    f = _features(view, j, cls, scores, cand, t_q, cfg.tau)
    p_on = 0.5
    if state_head is not None:
        p_on = float(state_head(view, j, cls))
    f["p_on"] = p_on
    return Answer("state", False, j, float(view.t[j]), state="on" if p_on >= 0.5 else "off", feats=f)


def answer(view: MemoryView, pq: ParsedQuery, cls: int, t_q: float, cfg: AnswerConfig, state_head=None) -> Answer:
    if pq.kind == "where":
        return answer_where(view, cls, t_q, cfg)
    if pq.kind == "temporal":
        return answer_temporal(view, cls, pq.verb, t_q, cfg)
    return answer_state(view, cls, t_q, cfg, state_head)
