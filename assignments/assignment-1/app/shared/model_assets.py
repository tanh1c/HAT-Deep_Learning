"""
Helpers for resolving demo model checkpoints from local paths or remote sources.

This module keeps deployment-specific file resolution out of the UI layer so the
same logic can be reused by Streamlit or any future lightweight demos.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DOWNLOAD_ROOT = Path(tempfile.gettempdir()) / "hat_dl_assignment1_models"
LOCAL_SECRET_CANDIDATES = (
    REPO_ROOT / ".streamlit" / "secrets.toml",
    REPO_ROOT / "deploy" / "streamlit-cloud" / ".streamlit" / "secrets.toml",
)


def load_local_secret_mapping() -> Mapping[str, Any]:
    """Load the nearest local secrets file if present."""
    for candidate in LOCAL_SECRET_CANDIDATES:
        if candidate.exists():
            with open(candidate, "rb") as f:
                data = tomllib.load(f)
            if isinstance(data, Mapping):
                return data
    return {}


def _mapping_get(mapping_like: Any, key: str) -> Any:
    """Safely read a key from a mapping-like object."""
    if mapping_like is None:
        return None

    if isinstance(mapping_like, Mapping):
        return mapping_like.get(key)

    getter = getattr(mapping_like, "get", None)
    if callable(getter):
        return getter(key)

    try:
        return mapping_like[key]
    except Exception:
        return None


def _lookup_secret(
    key: str,
    secrets: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Look up a config value from env vars or Streamlit secrets-like mappings."""
    env_value = os.getenv(key)
    if env_value:
        return env_value

    for secret_source in (load_local_secret_mapping(), secrets):
        if not secret_source:
            continue

        direct_value = _mapping_get(secret_source, key)
        if direct_value is not None:
            return str(direct_value)

        nested_models = _mapping_get(secret_source, "models")
        nested_value = _mapping_get(nested_models, key)
        if nested_value is not None:
            return str(nested_value)

    return None


def _resolve_configured_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _download_from_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request) as response, open(destination, "wb") as f:
        f.write(response.read())
    return destination


def _download_from_google_drive(identifier: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    import gdown

    if identifier.startswith("http://") or identifier.startswith("https://"):
        gdown.download(url=identifier, output=str(destination), fuzzy=True, quiet=False)
    else:
        gdown.download(id=identifier, output=str(destination), quiet=False)
    return destination


def ensure_model_checkpoint(
    *,
    filename: str,
    default_local_dir: Path,
    configured_path_key: str,
    configured_url_key: str,
    configured_gdrive_key: str,
    secrets: Optional[Mapping[str, Any]] = None,
    download_root: Path = DEFAULT_DOWNLOAD_ROOT,
) -> Tuple[Path, str]:
    """
    Resolve a checkpoint path from:
    1. configured absolute/relative path
    2. default local repo path
    3. cached downloaded file
    4. remote URL or Google Drive config

    Returns:
        (resolved_path, human_readable_source)
    """
    configured_path = _lookup_secret(configured_path_key, secrets)
    if configured_path:
        resolved_path = _resolve_configured_path(configured_path)
        if resolved_path.exists():
            return resolved_path, f"Configured local path ({configured_path_key})"
        raise FileNotFoundError(
            f"{configured_path_key} points to {resolved_path}, but that file does not exist."
        )

    default_local_path = default_local_dir / filename
    if default_local_path.exists():
        return default_local_path, "Repository local file"

    cached_path = download_root / filename
    if cached_path.exists():
        return cached_path, "Cached downloaded file"

    configured_gdrive = _lookup_secret(configured_gdrive_key, secrets)
    if configured_gdrive:
        downloaded = _download_from_google_drive(configured_gdrive, cached_path)
        return downloaded, f"Google Drive ({configured_gdrive_key})"

    configured_url = _lookup_secret(configured_url_key, secrets)
    if configured_url:
        downloaded = _download_from_url(configured_url, cached_path)
        return downloaded, f"Direct URL ({configured_url_key})"

    raise FileNotFoundError(
        "Model checkpoint could not be found locally and no remote source was configured. "
        f"Provide one of {configured_path_key}, {configured_gdrive_key}, or {configured_url_key}."
    )
