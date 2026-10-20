#!/usr/bin/env python3
"""Generate local paths shared by training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "paths.local.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure local SimLingo paths.")
    parser.add_argument("--root-path", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--carla-path", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Dataset root. Omit for inference-only setup.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = args.root_path.expanduser().resolve()
    carla_path = args.carla_path.expanduser().resolve()
    data_path = (
        (root_path / "data" / "simlingo").resolve()
        if args.data_path is None
        else args.data_path.expanduser().resolve()
    )

    train_entry = root_path / "start_train_adaption.py"
    required = [
        root_path / "start_eval_simlingo_adaption.py",
        carla_path / "CarlaUE4.sh",
        carla_path / "PythonAPI" / "carla",
    ]
    # 推理版仓库没有训练入口，因此不要求用户准备训练数据。
    if train_entry.is_file() and args.data_path is not None:
        required.extend(
            [
                data_path / "data" / "simlingo",
                data_path / "augmented_templates",
                data_path / "buckets_paths.pkl",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required paths are missing: {missing}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            {
                "root_path": str(root_path),
                "carla_path": str(carla_path),
                "data_path": str(data_path),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"Path config written to: {output}")


if __name__ == "__main__":
    main()
