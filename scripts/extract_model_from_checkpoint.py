"""
Load a trained distillation checkpoint using its config and checkpoint files.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_yaml
from lightning_module import DistillationLightningModule
from students.repvit.repvit_dino import RepVitDINO
from students.repvit.repvit_registry import REPVIT_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a distillation checkpoint with its config"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config used for training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file (.ckpt). If omitted, defaults to "
            "outputs/<experiment_name>/checkpoints/last.ckpt"
        ),
    )
    parser.add_argument(
        "--map-location",
        type=str,
        default="cpu",
        help="torch map_location value for checkpoint loading (default: cpu)",
    )
    return parser.parse_args()


def resolve_checkpoint_path(config: dict, checkpoint_arg: str | None) -> str:
    if checkpoint_arg is not None:
        checkpoint_path = checkpoint_arg
    else:
        experiment_name = config["logging"]["experiment_name"]
        checkpoint_path = os.path.join(
            "outputs", experiment_name, "checkpoints", "last.ckpt"
        )

    checkpoint_path = str(Path(checkpoint_path))
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def load_module_from_checkpoint(
    config_path: str, checkpoint_path: str | None = None, map_location: str = "cpu"
) -> DistillationLightningModule:
    config = load_yaml(config_path)
    resolved_checkpoint = resolve_checkpoint_path(config, checkpoint_path)

    module = DistillationLightningModule.load_from_checkpoint(
        checkpoint_path=resolved_checkpoint,
        config=config,
        map_location=map_location,
    )
    module.eval()
    return module


def resolve_output_path(config: dict) -> str:
    experiment_name = config["logging"]["experiment_name"]
    student_model_name = config["student"]["model"]
    output_filename = f"{student_model_name}_dino.pth"
    output_path = Path("outputs") / experiment_name / "checkpoints" / output_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def _get_student_adapter_weights(student_module: nn.Module) -> Tuple[bool, Optional[torch.Tensor]]:
    adapters = list(student_module.channel_adapters)
    if len(adapters) == 0:
        raise ValueError("Student has no channel adapters to export.")

    adapter = adapters[0]
    if isinstance(adapter, nn.Identity):
        return False, None

    if not hasattr(adapter, "projector"):
        raise ValueError(
            "Unsupported adapter format: expected adapter.projector in Student wrapper."
        )

    projector = adapter.projector
    if isinstance(projector, nn.Conv2d):
        return True, projector.weight.detach().cpu()

    raise ValueError(
        "Only 'basic' adapter export is supported for RepVitDINO (single Conv2d projector)."
    )


def export_student_as_repvit_dino(
    module: DistillationLightningModule,
    config: dict,
    output_path: str,
) -> None:
    student_cfg = config["student"]
    student_model_key = student_cfg["model"]

    if student_model_key not in REPVIT_MODELS:
        raise ValueError(
            f"Student model '{student_model_key}' not found in REPVIT_MODELS. "
            f"Available: {list(REPVIT_MODELS.keys())}"
        )

    timm_model_name = REPVIT_MODELS[student_model_key]["model_name"]

    use_adapter, adapter_weight = _get_student_adapter_weights(module.student)
    adapter_dim = int(adapter_weight.shape[0]) if adapter_weight is not None else None

    repvit_dino = RepVitDINO(
        backbone_name=timm_model_name,
        adapter_dim=adapter_dim,
        use_adapter=use_adapter,
    )

    student_backbone_sd = module.student.model.state_dict()

    missing, unexpected = repvit_dino.backbone.load_state_dict(
        student_backbone_sd,
        strict=False,
    )

    # Allow classifier/head keys to be missing for distilled feature backbones.
    non_head_missing = [k for k in missing if not k.startswith("head")]
    if non_head_missing or unexpected:
        raise RuntimeError(
            "Backbone key remap/load failed. "
            f"non_head_missing={non_head_missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    if use_adapter:
        if adapter_weight is None:
            raise ValueError("Adapter export expected weights but none were found.")
        if repvit_dino.adapter.weight.shape != adapter_weight.shape:
            raise ValueError(
                "Adapter weight shape mismatch: "
                f"checkpoint={tuple(adapter_weight.shape)}, "
                f"RepVitDINO={tuple(repvit_dino.adapter.weight.shape)}"
            )
        repvit_dino.adapter.weight.data.copy_(adapter_weight)

    torch.save(repvit_dino.state_dict(), output_path)


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)
    resolved_checkpoint = resolve_checkpoint_path(config, args.checkpoint)
    output_path = resolve_output_path(config)

    print(f"Loading config: {args.config}")
    print(f"Loading checkpoint: {resolved_checkpoint}")
    print(f"map_location: {args.map_location}")

    module = DistillationLightningModule.load_from_checkpoint(
        checkpoint_path=resolved_checkpoint,
        config=config,
        map_location=args.map_location,
    )
    module.eval()

    print("Checkpoint loaded successfully.")
    print(f"Teacher module: {module.teacher.__class__.__name__}")
    print(f"Student module: {module.student.__class__.__name__}")

    print("Exporting student weights as RepVitDINO (Student wrapper stripped)...")
    export_student_as_repvit_dino(module=module, config=config, output_path=output_path)
    print(f"Saved RepVitDINO weights: {output_path}")


if __name__ == "__main__":
    main()
