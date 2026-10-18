from pathlib import Path

from huggingface_hub import snapshot_download


def _download_repo(repo_id: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".complete"
    if marker.is_file():
        return target_dir
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir))
    marker.write_text(repo_id, encoding="utf-8")
    return target_dir


def ensure_weights(settings) -> dict[str, Path]:
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
    if volume is not None and hasattr(volume, "commit"):
        volume.commit()
