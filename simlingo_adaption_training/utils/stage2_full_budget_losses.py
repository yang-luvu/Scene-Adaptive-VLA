import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Stage2FullBudgetLosses:
    all_jsonl_path: Path
    measurement_to_loss_driving_full: Dict[str, float]
    num_samples: int
    mean_loss_driving_full: float


def _resolve_path(path_like: Any, workspace_root: Optional[Path]) -> Path:
    raw = str(path_like).strip()
    if raw == "":
        raise ValueError("path must be a non-empty string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        if workspace_root is None:
            raise ValueError(f"relative path requires workspace_root, got {path}")
        path = workspace_root / path
    return path.resolve()


def _extract_frame_id_from_measurement_path(measurement_path: Path) -> int:
    file_name = measurement_path.name
    if file_name.endswith(".json.gz"):
        file_name = file_name[: -len(".json.gz")]
    else:
        file_name = measurement_path.stem
    try:
        return int(file_name)
    except ValueError as exc:
        raise ValueError(
            f"failed to parse frame_id from measurement_path={measurement_path}"
        ) from exc


def _route_key_from_measurement_path(measurement_path: Path) -> Path:
    parts = str(measurement_path).rsplit("/measurements/", 1)
    if len(parts) == 2:
        return Path(parts[0]).resolve()
    return measurement_path.parent.resolve()


def _as_float(value: Any, field_name: str, line_no: int) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"line {line_no}: field '{field_name}' must be numeric, got bool"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_no}: field '{field_name}' must be numeric, got {value!r}"
        ) from exc


def _as_int(value: Any, field_name: str, line_no: int) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"line {line_no}: field '{field_name}' must be int, got bool"
        )
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"line {line_no}: field '{field_name}' must be int, got {value!r}"
        ) from exc


def _extract_loss_driving_full(row: Dict[str, Any], line_no: int) -> float:
    if "loss_driving_full" not in row:
        raise ValueError(f"line {line_no}: missing required field 'loss_driving_full'")
    return _as_float(row["loss_driving_full"], "loss_driving_full", line_no)


def load_and_validate_stage2_full_budget_losses(
    all_jsonl_path: str,
    workspace_root: Optional[Path] = None,
) -> Stage2FullBudgetLosses:
    root = None if workspace_root is None else Path(workspace_root).resolve()
    all_path = _resolve_path(all_jsonl_path, workspace_root=root)
    if not all_path.exists():
        raise ValueError(f"full-budget all.jsonl not found: {all_path}")
    if not all_path.is_file():
        raise ValueError(f"full-budget all.jsonl must be a file: {all_path}")
    if not all_path.name.endswith("_all.jsonl"):
        raise ValueError(
            f"full-budget loss file must end with '_all.jsonl', got {all_path.name}"
        )

    measurement_to_loss: Dict[str, float] = {}
    loss_driving_values = []

    with open(all_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid json") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: each row must be a JSON object")

            measurement_raw = row.get("measurement_path", None)
            if measurement_raw is None or str(measurement_raw).strip() == "":
                raise ValueError(f"line {line_no}: missing non-empty measurement_path")
            measurement_path = _resolve_path(measurement_raw, workspace_root=root)
            measurement_key = str(measurement_path)
            if measurement_key in measurement_to_loss:
                raise ValueError(
                    f"line {line_no}: duplicate measurement_path key after resolve(): {measurement_key}"
                )

            route_raw = row.get("route_key", None)
            if route_raw is None or str(route_raw).strip() == "":
                raise ValueError(f"line {line_no}: missing non-empty route_key")
            route_path = _resolve_path(route_raw, workspace_root=root)
            route_from_measurement = _route_key_from_measurement_path(measurement_path)
            if route_path != route_from_measurement:
                raise ValueError(
                    "line "
                    f"{line_no}: route_key mismatch with measurement_path after resolve(). "
                    f"route_key={route_path}, expected={route_from_measurement}"
                )

            frame_id = _as_int(row.get("frame_id", None), "frame_id", line_no)
            frame_from_measurement = _extract_frame_id_from_measurement_path(measurement_path)
            if frame_id != frame_from_measurement:
                raise ValueError(
                    "line "
                    f"{line_no}: frame_id mismatch with measurement_path. "
                    f"frame_id={frame_id}, expected={frame_from_measurement}, "
                    f"measurement_path={measurement_path}"
                )

            loss_driving_full = _extract_loss_driving_full(row, line_no)
            measurement_to_loss[measurement_key] = loss_driving_full
            loss_driving_values.append(loss_driving_full)

    if len(measurement_to_loss) == 0:
        raise ValueError(f"no valid rows found in {all_path}")

    mean_loss_driving_full = float(sum(loss_driving_values) / len(loss_driving_values))

    return Stage2FullBudgetLosses(
        all_jsonl_path=all_path,
        measurement_to_loss_driving_full=measurement_to_loss,
        num_samples=len(measurement_to_loss),
        mean_loss_driving_full=mean_loss_driving_full,
    )


def load_stage2_full_budget_losses_index_fast(
    all_jsonl_path: str,
    workspace_root: Optional[Path] = None,
) -> Stage2FullBudgetLosses:
    """
    Fast index-only loader used by DataModule at runtime.
    It keeps only the fields required for training lookup:
      measurement_path -> loss_driving_full
    """
    root = None if workspace_root is None else Path(workspace_root).resolve()
    all_path = _resolve_path(all_jsonl_path, workspace_root=root)
    if not all_path.exists():
        raise ValueError(f"full-budget all.jsonl not found: {all_path}")
    if not all_path.is_file():
        raise ValueError(f"full-budget all.jsonl must be a file: {all_path}")
    if not all_path.name.endswith("_all.jsonl"):
        raise ValueError(
            f"full-budget loss file must end with '_all.jsonl', got {all_path.name}"
        )

    measurement_to_loss: Dict[str, float] = {}
    loss_driving_values = []

    with open(all_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid json") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: each row must be a JSON object")

            measurement_raw = row.get("measurement_path", None)
            if measurement_raw is None or str(measurement_raw).strip() == "":
                raise ValueError(f"line {line_no}: missing non-empty measurement_path")
            measurement_path = _resolve_path(measurement_raw, workspace_root=root)
            measurement_key = str(measurement_path)
            if measurement_key in measurement_to_loss:
                raise ValueError(
                    f"line {line_no}: duplicate measurement_path key after resolve(): {measurement_key}"
                )

            loss_driving_full = _extract_loss_driving_full(row, line_no)
            measurement_to_loss[measurement_key] = loss_driving_full
            loss_driving_values.append(loss_driving_full)

    if len(measurement_to_loss) == 0:
        raise ValueError(f"no valid rows found in {all_path}")

    mean_loss_driving_full = float(sum(loss_driving_values) / len(loss_driving_values))

    return Stage2FullBudgetLosses(
        all_jsonl_path=all_path,
        measurement_to_loss_driving_full=measurement_to_loss,
        num_samples=len(measurement_to_loss),
        mean_loss_driving_full=mean_loss_driving_full,
    )
