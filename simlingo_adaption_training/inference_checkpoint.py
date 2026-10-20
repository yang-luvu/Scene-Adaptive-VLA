from typing import Dict

from torch import Tensor, nn


STAGE1_UNUSED_BUDGET_SCHEDULER_PREFIX = "budget_assigner.smart_budget_scheduler."


def load_policy_aware_inference_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Tensor],
    budget_policy: str,
) -> None:
    """Load inference weights while keeping all policy-relevant modules strict."""
    policy = str(budget_policy).strip().lower()
    if policy == "stage2":
        model.load_state_dict(state_dict, strict=True)
        return
    if policy != "stage1":
        raise ValueError(
            f"budget_policy must be one of ['stage1', 'stage2'], got {budget_policy!r}"
        )

    prefix = STAGE1_UNUSED_BUDGET_SCHEDULER_PREFIX
    expected_ignored_keys = {
        key for key in model.state_dict().keys() if key.startswith(prefix)
    }
    if not expected_ignored_keys:
        raise RuntimeError(
            f"Stage1 inference model exposes no parameters under {prefix!r}"
        )

    filtered_state_dict = {
        key: value for key, value in state_dict.items() if not key.startswith(prefix)
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    missing_keys = set(incompatible.missing_keys)
    unexpected_keys = set(incompatible.unexpected_keys)
    if missing_keys != expected_ignored_keys or unexpected_keys:
        missing_required = sorted(missing_keys - expected_ignored_keys)
        unexpectedly_loaded_scheduler = sorted(expected_ignored_keys - missing_keys)
        raise RuntimeError(
            "Stage1 inference checkpoint is incompatible outside the unused budget scheduler: "
            f"missing_required={missing_required}, "
            f"unexpected={sorted(unexpected_keys)}, "
            f"scheduler_keys_not_reported_missing={unexpectedly_loaded_scheduler}"
        )

    ignored_checkpoint_keys = sorted(
        key for key in state_dict.keys() if key.startswith(prefix)
    )
    print(
        "[Checkpoint] stage1 fixed-budget inference ignored the unused budget scheduler "
        f"({len(ignored_checkpoint_keys)} checkpoint keys, "
        f"{len(expected_ignored_keys)} current model keys)",
        flush=True,
    )
