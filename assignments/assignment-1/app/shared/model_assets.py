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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Optional, Tuple

import requests


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
    if not destination.exists() or destination.stat().st_size == 0:
        raise FileNotFoundError(
            f"Download from URL did not create a valid checkpoint file at {destination}."
        )
    return destination


class _GoogleDriveFormParser(HTMLParser):
    """Extract the confirmation form from a Google Drive warning page."""

    def __init__(self) -> None:
        super().__init__()
        self.form_action: Optional[str] = None
        self.inputs: dict[str, str] = {}
        self._inside_form = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        attr_map = dict(attrs)
        if tag == "form" and self.form_action is None:
            self.form_action = attr_map.get("action")
            self._inside_form = True
            return

        if tag == "input" and self._inside_form:
            name = attr_map.get("name")
            value = attr_map.get("value")
            if name and value is not None:
                self.inputs[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._inside_form:
            self._inside_form = False


def _is_valid_checkpoint_file(path: Path) -> bool:
    """Check whether a downloaded checkpoint looks like a PyTorch artifact."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    with open(path, "rb") as f:
        header = f.read(512)

    stripped = header.lstrip().lower()
    if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html"):
        return False

    return (
        header.startswith(b"PK\x03\x04")
        or header.startswith(b"PK\x05\x06")
        or header.startswith(b"PK\x07\x08")
        or header.startswith(b"\x80")
    )


def _write_response_to_destination(response: requests.Response, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return destination


def _download_google_drive_with_confirm(identifier: str, destination: Path) -> Path:
    session = requests.Session()
    initial_url = (
        identifier
        if identifier.startswith(("http://", "https://"))
        else f"https://drive.google.com/uc?id={identifier}&export=download"
    )
    response = session.get(initial_url, allow_redirects=True, stream=True, timeout=120)
    content_type = (response.headers.get("content-type") or "").lower()

    if "text/html" not in content_type:
        return _write_response_to_destination(response, destination)

    parser = _GoogleDriveFormParser()
    parser.feed(response.text)
    if not parser.form_action:
        raise RuntimeError("Google Drive returned HTML instead of a downloadable file.")

    confirmed = session.get(
        parser.form_action,
        params=parser.inputs,
        allow_redirects=True,
        stream=True,
        timeout=120,
    )
    return _write_response_to_destination(confirmed, destination)


def _download_from_google_drive(identifier: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        _download_google_drive_with_confirm(identifier, destination)
    except Exception:
        import gdown

        if identifier.startswith("http://") or identifier.startswith("https://"):
            gdown.download(url=identifier, output=str(destination), fuzzy=True, quiet=False)
        else:
            gdown.download(id=identifier, output=str(destination), quiet=False)

    if not _is_valid_checkpoint_file(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Google Drive download did not produce a valid checkpoint file. Expected: {destination}"
        )
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
        if _is_valid_checkpoint_file(cached_path):
            return cached_path, "Cached downloaded file"
        cached_path.unlink(missing_ok=True)

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
