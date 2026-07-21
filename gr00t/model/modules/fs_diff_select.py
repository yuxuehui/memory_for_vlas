"""TokenDrop-style pixel-difference keyframe selection for the framesamp inference cache.

Faithful frame-level port of RoboMME MME-VLA's TokenDrop scoring
(robomme_policy_learning/src/mme_vla_suite/shared/mem_buffer.py, _process_token_drop_score):
every `stride` policy calls, score the incoming frame by the mean |pixel difference| against
the LAST SCORED frame (their per-patch heap score, summed over the 8x8 grid, is proportional
to this frame-level mean), and keep a running top-(F-1) by score with frame 0 always kept
(their 1000.0 sentinel -> math.inf here). The read emits the selected keyframes in temporal
order plus the CURRENT frame, padded to exactly F frames by repeating the current frame —
mirroring both the FIFO cache's repeat-init and the training loader's fixed-F linspace layout,
so the downstream linspace->budget sub-sample sees the same token count as the FIFO path.

Motivation (Markdown/vlm_keyframe_labels probe, task_selectors.html): the inference-side
recent-FIFO framesamp window loses demo-phase/early events on long episodes, while
pixel-diff-selected frames cover the visual reference events (TD REF-R=1.0 vs FrameSamp-8
collapse). Training frames are episode-spanning; the FIFO was the outlier regime.
"""

from __future__ import annotations

import math

import torch


def split_fs_pixels(backbone_output, batch_size: int) -> torch.Tensor | None:
    """Per-sample flattened pixel signatures from the stashed `fs_pixels`, or None.

    `fs_pixels` is the raw eagle `pixel_values` (N, C, H, W) with N = B * views at K=1
    inference, stacked sample-major. Returns (B, views*C*H*W) or None when the tensor is
    missing or not evenly splittable (caller then falls back to feature-space scoring).
    """
    pix = backbone_output.get("fs_pixels", None) if hasattr(backbone_output, "get") else None
    if pix is None or not torch.is_tensor(pix) or pix.ndim < 2:
        return None
    if pix.shape[0] % batch_size != 0:
        return None
    return pix.reshape(batch_size, -1)


class DiffFrameSelector:
    """Per-session incremental top-K keyframe selection by pixel-difference score.

    State is bounded: at most (max_frames - 1) kept frames (token tensors), their scores and
    step indices, plus the last SCORED signature for the next diff. One instance per session;
    Gr00tPolicy round-trips instances across calls like the FIFO caches.
    """

    def __init__(self, max_frames: int, stride: int = 8):
        assert max_frames >= 2, "need at least frame-0 slot + current frame"
        self.max_frames = int(max_frames)
        self.stride = max(1, int(stride))
        self.reset()

    def reset(self) -> None:
        self.frames: list[torch.Tensor] = []  # (n_img, d) kept keyframe tokens
        self.scores: list[float] = []
        self.steps: list[int] = []
        self.last_sig: torch.Tensor | None = None
        self.step: int = -1

    def observe(self, vis: torch.Tensor, sig: torch.Tensor | None = None) -> None:
        """Advance one policy call with the current frame.

        vis: (n_img, d) raw image tokens of the current frame (what a kept slot stores).
        sig: flattened pixel signature for scoring; falls back to `vis` (feature-space
             diff) when pixels are unavailable — RoboMME found pixels work better, so the
             caller should pass them whenever it can.
        """
        self.step += 1
        if sig is None:
            sig = vis.reshape(-1)
        sig = sig.detach().float()

        if self.frames and self.frames[0].shape != vis.shape:
            # Defensive: token-count drift mid-session (batch-min n_img changed) — restart.
            self.reset()
            self.step = 0

        if self.step == 0:
            # Frame 0 is always kept (TokenDrop's 1000.0 sentinel on the first frame).
            self.frames = [vis.detach()]
            self.scores = [math.inf]
            self.steps = [0]
            self.last_sig = sig
            return

        if self.step % self.stride != 0:
            return

        score = (sig - self.last_sig).abs().mean().item()
        self.last_sig = sig

        cap = self.max_frames - 1  # one slot reserved for the current frame at read time
        if len(self.frames) < cap:
            self.frames.append(vis.detach())
            self.scores.append(score)
            self.steps.append(self.step)
        else:
            i_min = min(range(len(self.scores)), key=lambda i: self.scores[i])
            if score > self.scores[i_min]:  # frame 0 (inf) is never the argmin
                self.frames[i_min] = vis.detach()
                self.scores[i_min] = score
                self.steps[i_min] = self.step

    def read(self, vis_current: torch.Tensor) -> torch.Tensor:
        """Emit (max_frames * n_img, d): kept keyframes in temporal order + current frame,
        right-padded by repeating the current frame to exactly max_frames frames."""
        order = sorted(range(len(self.steps)), key=lambda i: self.steps[i])
        frames = [self.frames[i] for i in order]
        frames.append(vis_current)
        while len(frames) < self.max_frames:
            frames.append(vis_current)
        return torch.cat(frames[: self.max_frames], dim=0)
