import pickle
import struct

import pytest

from recall.data.loaders.hoi import PORTABLE, _SafeUnpickler, decode_frame


def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def _f(fn: int, x: float) -> bytes:
    return _varint(fn << 3 | 5) + struct.pack("<f", x)


def _ld(fn: int, b: bytes) -> bytes:
    return _varint(fn << 3 | 2) + _varint(len(b)) + b


def _v(fn: int, n: int) -> bytes:
    return _varint(fn << 3) + _varint(n)


def _frame() -> bytes:
    box = _f(1, 0.1) + _f(2, 0.2) + _f(3, 0.3) + _f(4, 0.4)
    hand = _ld(1, box) + _f(2, 0.9) + _v(3, PORTABLE) + _ld(4, _f(1, 0.05) + _f(2, -0.05)) + _v(5, 1)
    obj = _ld(1, _f(1, 0.2) + _f(2, 0.1) + _f(3, 0.4) + _f(4, 0.3)) + _f(2, 0.8)
    return _ld(1, b"P01_01") + _v(2, 7) + _ld(3, hand) + _ld(4, obj)


def test_decode_roundtrip():
    fr = decode_frame(_frame())
    assert fr.frame == 7
    h = fr.hands[0]
    assert h.side == 1 and h.state == PORTABLE
    assert h.box == pytest.approx((0.1, 0.2, 0.3, 0.4))
    o = fr.contacted_object(h)
    assert o is not None and o.score == pytest.approx(0.8)


def test_safe_unpickler_rejects_globals(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(print))
    with open(p, "rb") as f, pytest.raises(pickle.UnpicklingError):
        _SafeUnpickler(f).load()
    p.write_bytes(pickle.dumps([_frame()]))
    with open(p, "rb") as f:
        assert len(_SafeUnpickler(f).load()) == 1
