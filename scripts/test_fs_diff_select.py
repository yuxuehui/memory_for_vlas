"""Standalone smoke test for gr00t/model/modules/fs_diff_select.py (CPU, no weights).

Run:  .venv/bin/python scripts/test_fs_diff_select.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gr00t.model.modules.fs_diff_select import DiffFrameSelector, split_fs_pixels

N_IMG, D, F, STRIDE = 4, 8, 4, 8


def frame(step: int) -> torch.Tensor:
    # Encode the step index in the token values so kept frames are identifiable.
    return torch.full((N_IMG, D), float(step))


def pix(step: int) -> torch.Tensor:
    # Piecewise-constant pixel signal with step-function events at t=24 and t=56.
    base = torch.zeros(300)
    if step >= 24:
        base[:100] += 5.0
    if step >= 56:
        base[100:200] += 7.0
    return base


def kept_steps(sel, cur_step):
    out = sel.read(frame(cur_step))
    assert out.shape == (F * N_IMG, D), out.shape
    return [int(out[i * N_IMG, 0].item()) for i in range(F)]


# 1) Event frames survive, FIFO recency does not apply.
sel = DiffFrameSelector(max_frames=F, stride=STRIDE)
for t in range(100):
    sel.observe(frame(t), pix(t))
ks = kept_steps(sel, 99)
assert ks == [0, 24, 56, 99], ks  # frame 0 + the two event frames + current

# 2) Early episode: padded with the current frame.
sel2 = DiffFrameSelector(max_frames=F, stride=STRIDE)
sel2.observe(frame(0), pix(0))
ks2 = kept_steps(sel2, 0)
assert ks2 == [0, 0, 0, 0], ks2
sel2.observe(frame(1), pix(1))
ks2 = kept_steps(sel2, 1)
assert ks2 == [0, 1, 1, 1], ks2

# 3) Feature-space fallback (sig=None) runs and keeps the changed frame.
sel3 = DiffFrameSelector(max_frames=F, stride=STRIDE)
for t in range(40):
    v = frame(t) if t < 24 else frame(t) + 100.0  # feature jump at 24
    sel3.observe(v, None)
assert 24 in sel3.steps, sel3.steps

# 4) Eviction replaces the weakest, never frame 0.
sel4 = DiffFrameSelector(max_frames=F, stride=STRIDE)
for t in range(100):
    sel4.observe(frame(t), pix(t))
assert sel4.scores[0] == float("inf") and sel4.steps[0] == 0
assert len(sel4.frames) <= F - 1

# 5) split_fs_pixels shapes.
class _BO(dict):
    pass

bo = _BO(fs_pixels=torch.randn(6, 3, 8, 8))
assert split_fs_pixels(bo, 2).shape == (2, 3 * 3 * 8 * 8)
assert split_fs_pixels(bo, 4) is None  # not divisible
assert split_fs_pixels(_BO(), 2) is None

# 6) Reset clears state.
sel.reset()
sel.observe(frame(0), pix(0))
assert kept_steps(sel, 0) == [0, 0, 0, 0]

print("fs_diff_select: all checks passed")
