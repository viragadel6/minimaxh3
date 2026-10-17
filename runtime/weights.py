import fcntl
import json
import os
import threading
from pathlib import Path

from huggingface_hub import snapshot_download


_changed = False
_change_lock = threading.Lock()


def mark_changed():
    global _changed
    with _change_lock:
        _changed = True


def _download_snapshot(repo_id: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".complete"
    lock_path = directory.parent / f".{directory.name}.download.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if marker.is_file():
            try:
                metadata = json.loads(marker.read_text())
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if metadata.get("repo_id") == repo_id:
                return directory
            if metadata.get("repo_id"):
                raise ValueError(
                    f"{directory} belongs to another repository; use a new models volume"
                )

        snapshot_download(repo_id=repo_id, local_dir=str(directory))
        temporary = directory / ".complete.tmp"
        temporary.write_text(json.dumps({"repo_id": repo_id}, sort_keys=True))
        os.replace(temporary, marker)
        mark_changed()
    return directory


def ensure_weights(settings) -> dict[str, Path]:
    from runtime.lora import fused_specs, parse_lora, resolve_lora

    root = Path(settings.MODELS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "base": _download_snapshot(settings.MODEL_REPO_ID, root / "base")
    }
    if settings.TURBO_TRANSFORMER_REPO:
        paths["turbo"] = _download_snapshot(
            settings.TURBO_TRANSFORMER_REPO, root / "turbo"
        )
    for index, spec in enumerate(fused_specs(settings.FUSED_LORAS)):
        path = resolve_lora(spec.source, spec.weight_name)
        parse_lora(path, None)
        paths[f"lora:{index}"] = path
    return paths


def commit_if_changed(volume):
    global _changed
    with _change_lock:
        if _changed:
            volume.commit()
            _changed = False
