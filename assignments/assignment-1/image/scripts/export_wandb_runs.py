#!/usr/bin/env python
"""Export W&B project runs for Assignment 1 image experiments.

Usage examples:

    python assignments/assignment-1/image/scripts/export_wandb_runs.py \
        --entity nguyenquochieujff7-ho-chi-minh-city-university-of-technology \
        --project stanford-dogs-transfer-learning-new

    python assignments/assignment-1/image/scripts/export_wandb_runs.py \
        --entity nguyenquochieujff7-ho-chi-minh-city-university-of-technology \
        --project stanford-dogs-transfer-learning-new \
        --group stanford-dogs-fair-benchmark \
        --run-name-contains resnet_50 \
        --run-name-contains vit_b_16

Requirements:
- `wandb` installed
- `pandas` installed
- `wandb login` already completed on the machine that runs this script
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_output_dir = script_dir.parent / "artifacts" / "wandb_export"

    parser = argparse.ArgumentParser(
        description="Export W&B runs, histories, and run files for the Assignment 1 image experiments."
    )
    parser.add_argument("--entity", required=True, help="W&B entity or team slug.")
    parser.add_argument("--project", required=True, help="W&B project name.")
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Where exported CSVs, JSON files, and downloaded run files will be stored.",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="Optional W&B run group filter, for example 'stanford-dogs-fair-benchmark'.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Optional run ID filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--run-name-contains",
        action="append",
        default=[],
        help="Optional case-insensitive substring filter on run.name. Matches if any provided token appears in the run name.",
    )
    parser.add_argument(
        "--exclude-run-name-contains",
        action="append",
        default=[],
        help="Optional case-insensitive substring exclusion filter on run.name. Can be provided multiple times.",
    )
    parser.add_argument(
        "--file-pattern",
        default=None,
        help="Optional W&B file pattern for run.files(), for example 'media/%%' or '%.json'.",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Skip downloading run files/media and only export metadata + history CSVs.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Skip scan_history() export if you only want summary metadata or run files.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return value or "run"


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def normalize_value(value: Any) -> Any:
    safe_value = make_json_safe(value)
    if is_scalar(safe_value):
        return safe_value
    return json.dumps(safe_value, ensure_ascii=False, sort_keys=True)


def make_json_safe(value: Any) -> Any:
    if is_scalar(value):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(nested_value)
            for key, nested_value in value.items()
            if not (isinstance(key, str) and key.startswith("_"))
        }
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "items"):
        try:
            return {
                str(key): make_json_safe(nested_value)
                for key, nested_value in value.items()
                if not (isinstance(key, str) and key.startswith("_"))
            }
        except Exception:
            pass
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return [make_json_safe(item) for item in value]
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return make_json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def flatten_mapping(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        column = f"{prefix}{key}"
        if isinstance(value, dict):
            nested = flatten_mapping(value, f"{column}.")
            if nested:
                flat.update(nested)
            else:
                flat[column] = "{}"
        else:
            flat[column] = normalize_value(value)
    return flat


def run_matches_filters(run: Any, args: argparse.Namespace) -> bool:
    if args.group and getattr(run, "group", None) != args.group:
        return False

    if args.run_id and run.id not in set(args.run_id):
        return False

    name = (run.name or "").lower()
    if args.run_name_contains:
        if not any(token.lower() in name for token in args.run_name_contains):
            return False

    if args.exclude_run_name_contains:
        if any(token.lower() in name for token in args.exclude_run_name_contains):
            return False

    return True


def export_run_summary(run: Any) -> dict[str, Any]:
    summary = make_json_safe(dict(run.summary))
    config = make_json_safe(dict(run.config))

    base = {
        "run_id": run.id,
        "run_name": run.name,
        "display_name": run.display_name,
        "url": run.url,
        "state": run.state,
        "entity": run.entity,
        "project": run.project,
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "created_at": run.created_at,
        "updated_at": getattr(run, "updated_at", None),
    }
    base.update(flatten_mapping(config, "config."))
    base.update(flatten_mapping(summary, "summary."))
    return base


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def export_run_history(run: Any, run_dir: Path) -> Path:
    history_records = list(run.scan_history())
    history_df = pd.DataFrame(history_records)
    history_path = run_dir / "history.csv"
    history_df.to_csv(history_path, index=False)
    return history_path


def download_run_files(run: Any, run_dir: Path, pattern: str | None) -> list[dict[str, Any]]:
    files_dir = run_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict[str, Any]] = []
    for file_obj in run.files(pattern=pattern):
        file_obj.download(root=files_dir, replace=True)
        downloaded.append(
            {
                "name": file_obj.name,
                "url": getattr(file_obj, "url", None),
                "size_bytes": getattr(file_obj, "size", None) or getattr(file_obj, "sizeBytes", None),
                "updated_at": getattr(file_obj, "updated_at", None),
            }
        )
    return downloaded


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Using the existing local W&B login session. Run `wandb login` first if this machine is not authenticated yet.")
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    print(f"Querying W&B project: {project_path}")

    matched_runs = [run for run in api.runs(project_path) if run_matches_filters(run, args)]
    if not matched_runs:
        print("No runs matched the provided filters.")
        return 1

    print(f"Matched runs: {len(matched_runs)}")
    summary_rows: list[dict[str, Any]] = []
    manifest_runs: list[dict[str, Any]] = []

    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for index, run in enumerate(matched_runs, start=1):
        run_label = f"{slugify(run.name or run.id)}__{run.id}"
        run_dir = runs_root / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(matched_runs)}] Exporting run: {run.name} ({run.id})")
        summary_row = export_run_summary(run)
        summary_rows.append(summary_row)

        raw_metadata = {
            "id": run.id,
            "name": run.name,
            "display_name": run.display_name,
            "url": run.url,
            "state": run.state,
            "entity": run.entity,
            "project": run.project,
            "group": getattr(run, "group", None),
            "job_type": getattr(run, "job_type", None),
            "created_at": run.created_at,
            "updated_at": getattr(run, "updated_at", None),
            "config": make_json_safe(dict(run.config)),
            "summary": make_json_safe(dict(run.summary)),
        }
        write_json(run_dir / "run_metadata.json", raw_metadata)

        run_manifest: dict[str, Any] = {
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
            "metadata_json": str((run_dir / "run_metadata.json").relative_to(output_dir)),
        }

        if not args.skip_history:
            history_path = export_run_history(run, run_dir)
            print(f"  saved history -> {history_path}")
            run_manifest["history_csv"] = str(history_path.relative_to(output_dir))

        if not args.skip_files:
            downloaded_files = download_run_files(run, run_dir, args.file_pattern)
            files_manifest_path = run_dir / "files_manifest.json"
            write_json(files_manifest_path, downloaded_files)
            print(f"  downloaded files -> {len(downloaded_files)}")
            run_manifest["files_manifest_json"] = str(files_manifest_path.relative_to(output_dir))
            run_manifest["downloaded_file_count"] = len(downloaded_files)

        manifest_runs.append(run_manifest)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "runs_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV -> {summary_csv_path}")

    write_json(
        output_dir / "export_manifest.json",
        {
            "entity": args.entity,
            "project": args.project,
            "group": args.group,
            "filters": {
                "run_id": args.run_id,
                "run_name_contains": args.run_name_contains,
                "exclude_run_name_contains": args.exclude_run_name_contains,
                "file_pattern": args.file_pattern,
                "skip_files": args.skip_files,
                "skip_history": args.skip_history,
            },
            "runs": manifest_runs,
        },
    )
    print(f"Saved export manifest -> {output_dir / 'export_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
