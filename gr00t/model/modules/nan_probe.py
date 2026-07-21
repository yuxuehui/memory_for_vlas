# SPDX-License-Identifier: Apache-2.0
"""Lightweight NaN/Inf source locator — finds the FIRST module that emits a non-finite
tensor in the forward OR backward pass, without the ~10x cost of autograd anomaly mode.

Use: set env DEBUG_NAN=1 in the launch; launch_finetune calls install_nan_probe(model).
On the first non-finite output/grad it prints the module's qualified name + tensor stats and
(for backward) whether its INPUTS were finite — i.e. whether THIS module created the NaN or
merely propagated it. Fires once per step (deduped) so an intermittent NaN-batch is captured live.
"""
from __future__ import annotations

import torch


def _minmax(t):
    # COPY-FREE: amax/amin are reductions (no full-size intermediate, unlike .float()/.abs()).
    # amax/amin PROPAGATE NaN and surface +-Inf -> catches both without overflow or cancellation.
    return float(t.amin()), float(t.amax())


def _finite(t):
    lo, hi = _minmax(t)
    import math
    return math.isfinite(lo) and math.isfinite(hi)


def _stats(t):
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return "n/a"
    lo, hi = _minmax(t)
    return f"shape={tuple(t.shape)} dtype={t.dtype} min={lo:.3e} max={hi:.3e}"


def _bad(t):
    return isinstance(t, torch.Tensor) and t.is_floating_point() and not _finite(t)


def _any_bad(xs):
    return any(_bad(x) for x in xs if isinstance(x, torch.Tensor))


def install_nan_probe(model, max_reports: int = 40):
    state = {"n": 0, "fwd_seen": False, "bwd_seen": False, "aborted": False}

    def _dump_params(mod):
        lines = []
        for pn, p in mod.named_parameters(recurse=False):
            lines.append(f"        param {pn}: {_stats(p.data)}")
        return "\n".join(lines)

    def _dump_inputs(inputs):
        lines = []
        for i, x in enumerate(inputs):
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                lines.append(f"        in[{i}]: {_stats(x)}")
        return "\n".join(lines)

    def fwd_hook(name):
        def hook(mod, inputs, output):
            if state["n"] >= max_reports:
                return
            outs = output if isinstance(output, (tuple, list)) else (output,)
            if _any_bad(outs):
                ins_ok = not _any_bad([x for x in inputs])
                origin = "ORIGIN (inputs finite)" if ins_ok else "propagated (bad input)"
                bad_out = next(o for o in outs if _bad(o))
                print(f"[NAN-PROBE][FWD] {origin} module={name} ({mod.__class__.__name__})\n"
                      f"        out: {_stats(bad_out)}\n{_dump_inputs(inputs)}\n{_dump_params(mod)}",
                      flush=True)
                state["n"] += 1
                if ins_ok and not state["aborted"]:
                    state["aborted"] = True
                    raise RuntimeError(f"NAN-PROBE: first forward NaN ORIGIN at {name} — aborting to capture state")
        return hook

    def bwd_hook(name):
        def hook(mod, grad_input, grad_output):
            if state["n"] >= max_reports:
                return
            gi = [g for g in (grad_input or []) if isinstance(g, torch.Tensor)]
            go = [g for g in (grad_output or []) if isinstance(g, torch.Tensor)]
            if _any_bad(gi):
                go_ok = not _any_bad(go)
                origin = "ORIGIN (grad_output finite)" if go_ok else "propagated (bad grad_output)"
                bad_gi = next(g for g in gi if _bad(g))
                go_stats = "\n".join(f"        grad_out[{i}]: {_stats(g)}" for i, g in enumerate(go))
                print(f"[NAN-PROBE][BWD] {origin} module={name} ({mod.__class__.__name__})\n"
                      f"        grad_in: {_stats(bad_gi)}\n{go_stats}\n{_dump_params(mod)}", flush=True)
                state["n"] += 1
        return hook

    # PARAM-GRADIENT hook: fires the instant a param's .grad is accumulated in the backward —
    # catches the TRUE origin (first NaN param-grad) before the optimizer corrupts weights and
    # before the next forward hides it. Full speed (no anomaly 10x cost). THE tool for this NaN.
    def grad_hook(name):
        def hook(p):
            if state["n"] >= max_reports:
                return
            g = p.grad
            if g is not None and _bad(g):
                print(f"[NAN-PROBE][GRAD] FIRST NaN param-grad: param={name}\n"
                      f"        grad: {_stats(g)}\n        param: {_stats(p.data)}", flush=True)
                state["n"] += 1
        return hook

    ng = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(grad_hook(name))
            ng += 1
    print(f"[NAN-PROBE] installed param-grad hooks on {ng} trainable params", flush=True)

    n = 0
    for name, mod in model.named_modules():
        # Hook only modules that OWN a trainable param (the integration path: backbone top layers,
        # moment_tokens, the mamba memory, the DiT action head) — where the mamba NaN originates.
        # This cuts the frozen-backbone bulk (memory + noise) that caused the earlier OOM.
        if not any(p.requires_grad for p in mod.parameters(recurse=False)):
            continue
        mod.register_forward_hook(fwd_hook(name or "<root>"))
        mod.register_full_backward_hook(bwd_hook(name or "<root>"))
        n += 1
    print(f"[NAN-PROBE] installed forward+backward hooks on {n} trainable modules (max_reports={max_reports})", flush=True)
    return state
