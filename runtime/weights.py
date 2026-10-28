"""Weight provisioning for the Modal container.

Downloads are idempotent and atomic: a marker file is written only after
``snapshot_download`` succeeded, so an interrupted download is retried instead of
being treated as complete.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_MARKER = ".complete"
_ALLOW_PATTERNS = ("*.safetensors", "*.json", "*.txt", "*.model")


def _download_repo(repo_id: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / _MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == repo_id:
        return target_dir
    logger.info("downloading %s -> %s", repo_id, target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        allow_patterns=list(_ALLOW_PATTERNS),
    )
    # Written last: a crashed download never leaves a valid marker behind.
    tmp_marker = marker.with_suffix(".tmp")
    tmp_marker.write_text(repo_id, encoding="utf-8")
    os.replace(tmp_marker, marker)
    return target_dir


def ensure_weights(settings) -> dict[str, Path]:
    """Make sure every weight tree the server needs exists locally."""
    from runtime.lora import parse_fused_loras, resolve_lora

    models_root = Path(settings.MODELS_DIR)
    models_root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {
        "base": _download_repo(settings.MODEL_REPO_ID, models_root / "base"),
    }
    if settings.TURBO_TRANSFORMER_REPO:
        paths["turbo"] = _download_repo(
            settings.TURBO_TRANSFORMER_REPO, models_root / "turbo"
        )
    for idx, spec in enumerate(parse_fused_loras(settings.FUSED_LORAS)):
        paths[f"lora_{idx}"] = resolve_lora(spec.source, spec.weight_name)
    return paths


def commit_if_changed(volume) -> None:
    """Commit ``volume`` when it exposes Modal's commit API (no-op for stubs)."""
    commit = getattr(volume, "commit", None)
    if callable(commit):
        commit()
