from dataclasses import dataclass
import json
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""


def _install_attn_capture(policy, attn_dir):
    """ENV-GATED (ATTN_DUMP_DIR): dump per policy call the action->image attention heatmap per DiT
    cross-attn layer for BOTH camera views (front=view0, wrist=view1) + both frames + img/lang split.
    Closed-loop rollout introspection. (Separate server file so the normal eval server stays pristine.)"""
    import sys
    import numpy as np
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    import visualize_memory_attention as V  # noqa: E402

    inner = getattr(policy, "policy", policy)
    os.makedirs(attn_dir, exist_ok=True)
    V.install_attn_hooks()
    V.install_imagemask_hook(inner.model)
    n_cross = int(inner.model.config.diffusion_model_cfg["num_layers"]) // 2
    vkeys = inner.modality_configs["video"].modality_keys
    st = {"n": 0}
    orig = policy.get_action

    def _frame(observation, key):
        fv = np.asarray(observation.get(f"video.{key}", observation.get(key)))
        while fv.ndim > 3:
            fv = fv[0]
        return fv.astype(np.uint8)

    def wrapped(observation, *a, **kw):
        V._CAP["cross"].clear(); V._CAP["image_mask"] = None
        out = orig(observation, *a, **kw)
        try:
            front = _frame(observation, vkeys[0])
            wrist = _frame(observation, vkeys[1]) if len(vkeys) > 1 else front
            im = V._CAP["image_mask"]; raw = list(V._CAP["cross"])
            if im is not None and raw:
                mask = im[0].bool()
                n_den = max(1, len(raw) // n_cross)
                fronts, wrists, imgs, langs = [], [], [], []
                for li in range(n_cross):
                    As = [raw[d * n_cross + li] for d in range(n_den) if d * n_cross + li < len(raw)]
                    av = sum(A[0].float().mean(0) for A in As) / len(As)
                    if av.shape[0] != mask.shape[0]:
                        continue
                    fronts.append(V.img_heat_from_cross(av[mask], n_views=2, view=0))
                    wrists.append(V.img_heat_from_cross(av[mask], n_views=2, view=1))
                    imgs.append(float(av[mask].sum())); langs.append(float(av[~mask].sum()))
                zz = lambda L: np.asarray([(h if h is not None else np.zeros((9, 9), np.float32)) for h in L], np.float32)
                np.savez_compressed(os.path.join(attn_dir, f"call_{st['n']:04d}.npz"),
                                    front=front, wrist=wrist,
                                    front_heats=zz(fronts), wrist_heats=zz(wrists), heats=zz(fronts),
                                    img=np.asarray(imgs, np.float32), lang=np.asarray(langs, np.float32))
                st["n"] += 1
        except Exception as e:
            print(f"[attn-capture] err: {type(e).__name__}: {e}")
        return out

    policy.get_action = wrapped
    print(f"[attn-capture] ENABLED(2-view) -> {attn_dir} (n_cross={n_cross}, vkeys={vkeys})")


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    _attn_dir = os.environ.get("ATTN_DUMP_DIR")
    if _attn_dir and config.use_sim_policy_wrapper:
        _install_attn_capture(policy, _attn_dir)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
