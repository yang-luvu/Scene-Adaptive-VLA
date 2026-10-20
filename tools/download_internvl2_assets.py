#!/usr/bin/env python3
"""Download InternVL2-1B data assets without replacing local model code."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = (
    PROJECT_ROOT
    / "simlingo_adaption_training"
    / "models"
    / "language_model"
    / "internvl_2_1b"
)
CUSTOM_MODEL_FILES = (
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "configuration_qwen2.py",
    "conversation.py",
    "modeling_flash_attention_utils.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
    "modeling_qwen2.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download InternVL2-1B weights and tokenizer/config assets only."
    )
    parser.add_argument("--repo-id", default="OpenGVLab/InternVL2-1B")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target.expanduser().resolve()
    missing_code = [name for name in CUSTOM_MODEL_FILES if not (target / name).is_file()]
    if missing_code:
        raise FileNotFoundError(
            f"Custom InternVL2 source files are missing from {target}: {missing_code}"
        )

    # Download assets without overwriting the custom dynamic-layer Python code.
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(target),
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    print(f"InternVL2 assets are ready at: {target}")


if __name__ == "__main__":
    main()
