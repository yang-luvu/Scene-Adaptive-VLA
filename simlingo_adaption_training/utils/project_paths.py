from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.local.yaml"


@dataclass(frozen=True)
class ProjectPaths:
    """Own machine-specific roots and derive all other project paths."""

    root_path: Path
    carla_path: Path
    data_path: Path

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ProjectPaths":
        path = (
            DEFAULT_PATHS_CONFIG
            if config_path is None
            else Path(config_path).expanduser().resolve()
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Path config not found: {path}. "
                "Run tools/setup_paths.py before training or evaluation."
            )

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        missing = [key for key in ("root_path", "carla_path", "data_path") if not data.get(key)]
        if missing:
            raise ValueError(f"Missing path config fields {missing}: {path}")

        return cls(
            root_path=Path(data["root_path"]).expanduser().resolve(),
            carla_path=Path(data["carla_path"]).expanduser().resolve(),
            data_path=Path(data["data_path"]).expanduser().resolve(),
        )

    def resolve_root(self, path_like: str) -> Path:
        path = Path(path_like).expanduser()
        return path.resolve() if path.is_absolute() else (self.root_path / path).resolve()

    def resolve_data(self, path_like: str) -> Path:
        path = Path(path_like).expanduser()
        return path.resolve() if path.is_absolute() else (self.data_path / path).resolve()

    @property
    def bench2drive_path(self) -> Path:
        return self.root_path / "Bench2Drive"

    @property
    def leaderboard_path(self) -> Path:
        return self.bench2drive_path / "leaderboard"

    @property
    def scenario_runner_path(self) -> Path:
        return self.bench2drive_path / "scenario_runner"

    @property
    def route_path(self) -> Path:
        return self.root_path / "leaderboard" / "data" / "bench2drive_split"

    @property
    def agent_file(self) -> Path:
        return self.root_path / "team_code_adaption" / "agent_simlingo.py"

    @property
    def carla_python_api(self) -> Path:
        return self.carla_path / "PythonAPI" / "carla"

    @property
    def carla_egg(self) -> Path:
        pattern = f"carla-0.9.15-py{sys.version_info.major}.*.egg"
        candidates = sorted((self.carla_python_api / "dist").glob(pattern))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected one CARLA egg matching {pattern} under "
                f"{self.carla_python_api / 'dist'}, found={candidates}"
            )
        return candidates[0]

    @property
    def template_root(self) -> Path:
        return self.data_path / "augmented_templates"

    @property
    def measurement_root(self) -> Path:
        return self.data_path / "data" / "simlingo"

    @property
    def commentary_root(self) -> Path:
        return self.data_path / "commentary" / "simlingo"

    @property
    def vqa_root(self) -> Path:
        return self.data_path / "drivelm" / "simlingo"

    @property
    def route_stream_cache(self) -> Path:
        return self.data_path / "route_stream_indices"
