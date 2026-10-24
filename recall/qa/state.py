"""On/off state head for appliance queries ("Did I turn off the tap?").

Trained on train-split touch events (or frames, for B2) aligned with
turn-on / turn-off narrations. Input = [location embedding, scene embedding]
of the last interaction with the appliance; output P(on). Falls back to the
per-appliance training prior when the probe is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from recall.data.queries import STATE_KEYS, VERB_TURN_OFF, VERB_TURN_ON
from recall.qa.answer import MemoryView


@dataclass
class StateHead:
    clf: LogisticRegression | None
    prior_on: dict[int, float] = field(default_factory=dict)
    global_prior: float = 0.5

    def features(self, view: MemoryView, j: int) -> np.ndarray:
        return np.concatenate([view.loc_emb[j], view.scene_emb[j]])

    def __call__(self, view: MemoryView, j: int, cls: int) -> float:
        prior = self.prior_on.get(cls, self.global_prior)
        if self.clf is None:
            return prior
        return float(self.clf.predict_proba(self.features(view, j)[None])[0, 1])


def state_training_pairs(
    narr,
    views: dict[str, MemoryView],
    key2id: dict[str, int],
    window_s: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = {key2id[k] for k in STATE_KEYS if k in key2id}
    X, y, c = [], [], []
    for vid, view in views.items():
        d = narr[
            (narr.video_id == vid) & narr.verb_class.isin([VERB_TURN_ON, VERB_TURN_OFF]) & narr.noun_class.isin(ids)
        ]
        for r in d.itertuples():
            m = (view.t >= r.start_s - window_s) & (view.t <= r.stop_s + window_s)
            m &= np.isin(view.kind, ("touch", "take", "frame"))
            js = np.flatnonzero(m)
            if len(js) == 0:
                continue
            j = int(js[-1])
            X.append(np.concatenate([view.loc_emb[j], view.scene_emb[j]]))
            y.append(int(r.verb_class == VERB_TURN_ON))
            c.append(int(r.noun_class))
    D = 2 * (next(iter(views.values())).loc_emb.shape[1] if views else 1)
    return (np.array(X, np.float32).reshape(-1, D), np.array(y, int), np.array(c, int))


def fit_state_head(X: np.ndarray, y: np.ndarray, cls: np.ndarray, C: float = 1.0) -> StateHead:
    prior = {int(k): float(y[cls == k].mean()) for k in np.unique(cls)} if len(y) else {}
    gp = float(y.mean()) if len(y) else 0.5
    clf = None
    if len(y) >= 20 and 0 < y.mean() < 1:
        clf = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    return StateHead(clf, prior, gp)
