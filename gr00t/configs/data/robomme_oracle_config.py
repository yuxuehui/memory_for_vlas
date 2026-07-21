# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modality config for the RoboMME ORACLE-TEXT experiment (memoryless upper bound).
# Identical to robomme_config.py EXCEPT the language modality reads the per-episode
# instruction ("task" -> meta/episodes.jsonl tasks), which in the oracle dataset
# (data/robomme_oracle) holds the demo's ground-truth fact written into the text.
# (The default robomme_config reads "annotation.human.action.task_description", whose
# parquet task_index still points at the ORIGINAL 116 strings — wrong for the oracle
# set, since the oracle text is per-episode and lives in episodes.jsonl.)

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


robomme_panda_joint_oracle = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front_view", "wrist_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_position", "gripper_position"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["joint_position", "gripper_close"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["task"],   # <-- oracle text per episode (episodes.jsonl)
    ),
}


register_modality_config(robomme_panda_joint_oracle, EmbodimentTag.NEW_EMBODIMENT)
