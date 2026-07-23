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

    mem_fs_select: str | None = None
    """Override inference framesamp frame selection on the loaded checkpoint:
    'fifo' (recent-F window, default behavior) or 'diff' (TokenDrop-style
    pixel-difference keyframes). None = use the checkpoint config."""

    mem_fs_diff_stride: int | None = None
    """Override the 'diff'/'patch_union' scoring cadence in policy calls (default 8)."""

    mem_fs_attn_layer: int | None = None
    """patch_union: DiT cross-attn layer for the attention channel (default 13)."""

    mem_fs_tail_share: float | None = None
    """patch_union: tail_L15 channel share of the non-novelty budget (default 0.0 = 2-way)."""

    mem_fs_pos_rope: bool | None = None
    """patch_union: PPE-style 3D key RoPE on memory tokens (default from checkpoint)."""


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
        # Eval-time framesamp keyframe-selection override (inference-only knob; safe on
        # any checkpoint — the model reads it via getattr with a "fifo" default).
        if config.mem_fs_select is not None:
            assert config.mem_fs_select in ("fifo", "diff", "patch_union"), config.mem_fs_select
            policy.model.action_head.mem_fs_select = config.mem_fs_select
            print(f"  mem_fs_select override: {config.mem_fs_select}")
        if config.mem_fs_diff_stride is not None:
            policy.model.action_head.mem_fs_diff_stride = int(config.mem_fs_diff_stride)
            print(f"  mem_fs_diff_stride override: {config.mem_fs_diff_stride}")
        if config.mem_fs_attn_layer is not None:
            policy.model.action_head.mem_fs_attn_layer = int(config.mem_fs_attn_layer)
            print(f"  mem_fs_attn_layer override: {config.mem_fs_attn_layer}")
        if config.mem_fs_tail_share is not None:
            policy.model.action_head.mem_fs_tail_share = float(config.mem_fs_tail_share)
            print(f"  mem_fs_tail_share override: {config.mem_fs_tail_share}")
        if config.mem_fs_pos_rope is not None:
            policy.model.action_head.mem_fs_pos_rope = bool(config.mem_fs_pos_rope)
            print(f"  mem_fs_pos_rope override: {config.mem_fs_pos_rope}")
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
