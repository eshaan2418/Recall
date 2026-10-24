"""Episodic event schema (Contribution 2): what Recall stores instead of video."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVENT_KINDS = ("put", "take", "touch", "frame")  # "frame" = B2 baseline pseudo-events


@dataclass
class Event:
    """One remembered hand–object interaction.

    ``obj_emb`` is the mean embedding of the held/touched object crops;
    ``loc_emb`` describes where it ended up (surroundings at release);
    ``scene_emb`` the whole frame at the event (used for place clustering).
    Probabilities are filled by :mod:`recall.perception.heads`.
    """

    stream_id: str
    t: float  # seconds since stream start
    kind: str  # put | take | touch
    side: int
    frame_idx: int  # index into the 2 fps timeline
    obj_emb: np.ndarray
    loc_emb: np.ndarray
    scene_emb: np.ndarray
    n_obs: int = 1  # processed frames supporting this event
    hand_score: float = 0.0
    release_observed: bool = False  # True if the frame right after release was processed
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    t_start: float = 0.0
    obj_top: list[tuple[int, float]] = field(default_factory=list)  # (noun class, prob)
    loc_top: list[tuple[int, float]] = field(default_factory=list)  # (location class, prob)
    place_id: int = -1
    event_id: int = -1
    thumb: bytes | None = None

    def __post_init__(self) -> None:
        assert self.kind in EVENT_KINDS, self.kind
