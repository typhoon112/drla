"""Small SwanLab tracking helpers for DRLA experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_TRACKING_CONFIG = Path("/data1/luyifei/drla/configs/swanlab.yaml")


def load_tracking_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load SwanLab defaults from a YAML file."""
    config_path = Path(path) if path else DEFAULT_TRACKING_CONFIG
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Tracking config must be a mapping: {config_path}")
    return loaded


def init_experiment(
    *,
    stage: str,
    config: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    experiment_name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: str | None = None,
    resume_id: str | None = None,
    tracking_config: str | os.PathLike[str] | None = None,
    **kwargs: Any,
):
    """Initialize a SwanLab run using DRLA-wide defaults.

    Environment overrides:
      SWANLAB_PROJECT, SWANLAB_WORKSPACE, SWANLAB_MODE, SWANLAB_EXPERIMENT_NAME,
      SWANLAB_GROUP, SWANLAB_RUN_ID, SWANLAB_RESUME, SWANLAB_TAGS.
    """
    import swanlab

    defaults = load_tracking_config(tracking_config)
    stage_defaults = defaults.get("stages", {}).get(stage, {})
    default_tags = list(defaults.get("default_tags", []))
    stage_tags = list(stage_defaults.get("tags", []))
    env_tags = _split_tags(os.getenv("SWANLAB_TAGS"))

    run_tags = _dedupe(default_tags + stage_tags + (tags or []) + env_tags)
    run_group = os.getenv("SWANLAB_GROUP") or group or stage_defaults.get("group") or stage
    env_mode = os.getenv("SWANLAB_MODE")
    run_mode = mode if mode is not None else None if env_mode else defaults.get("mode", "cloud")
    run_id = resume_id or os.getenv("SWANLAB_RUN_ID")
    resume = os.getenv("SWANLAB_RESUME") or ("allow" if run_id else None)

    return swanlab.init(
        project=os.getenv("SWANLAB_PROJECT") or defaults.get("project", "drla-mvp"),
        workspace=os.getenv("SWANLAB_WORKSPACE") or defaults.get("workspace"),
        experiment_name=(
            os.getenv("SWANLAB_EXPERIMENT_NAME")
            or experiment_name
            or stage_defaults.get("experiment_name")
        ),
        description=description or stage_defaults.get("description"),
        job_type=job_type or stage_defaults.get("job_type") or stage,
        group=run_group,
        tags=run_tags,
        config=config,
        logdir=defaults.get("logdir", "/data1/luyifei/drla/outputs/swanlog"),
        mode=run_mode,
        id=run_id,
        resume=resume,
        **kwargs,
    )


def require_swanlab_disabled_for_non_training(mode: str, *, script_kind: str) -> None:
    """Reject SwanLab logging for scripts that do not update model weights."""
    if mode != "disabled":
        raise ValueError(
            f"{script_kind} has no optimizer/backward training loop and must not use SwanLab; "
            "run it with --swanlab-mode disabled and keep outputs as local artifacts."
        )


def log_metrics(
    metrics: Mapping[str, Any],
    *,
    step: int | None = None,
    prefix: str | None = None,
    print_to_console: bool = False,
) -> None:
    """Log scalar metrics, optionally namespaced by a prefix."""
    import swanlab

    payload = {
        f"{prefix}/{key}" if prefix else key: value for key, value in metrics.items()
    }
    swanlab.log(payload, step=step, print_to_console=print_to_console)


def finish_experiment() -> None:
    """Explicitly close the active SwanLab run."""
    import swanlab

    swanlab.finish()


def _split_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
