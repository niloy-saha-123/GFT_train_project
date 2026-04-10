#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rail_video_intelligence.pipeline import (
    RailVideoPipeline,
    SheetsConfig,
    load_camera_profile,
    load_pipeline_settings,
    parse_pipeline_settings,
    summarize_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rail Video Ops v1 pipeline runner.")
    parser.add_argument("--pipeline-config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--camera-profile", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Single video source path.")
    parser.add_argument("--source-dir", type=Path, help="Directory of videos for batch processing.")
    parser.add_argument("--run-name", default="v1_run")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v1"))
    parser.add_argument("--delete-on-success", action="store_true")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--show-live", action=argparse.BooleanOptionalAction, default=None)
    return parser


def discover_videos(source_dir: Path) -> List[Path]:
    extensions = {".mp4", ".mov", ".mkv", ".avi"}
    return sorted([path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions])


def main() -> None:
    args = build_parser().parse_args()
    if not args.source and not args.source_dir:
        raise SystemExit("Provide --source or --source-dir.")

    raw = load_pipeline_settings(args.pipeline_config)
    settings = parse_pipeline_settings(raw.get("pipeline", raw))
    sheets_raw = raw.get("sheets", {})
    sheets_config = SheetsConfig(
        enabled=bool(sheets_raw.get("enabled", False)),
        spreadsheet_id=sheets_raw.get("spreadsheet_id"),
        worksheet_name=sheets_raw.get("worksheet_name", "events"),
        credentials_path=Path(sheets_raw["credentials_path"]) if sheets_raw.get("credentials_path") else None,
        retries=int(sheets_raw.get("retries", 4)),
        backoff_seconds=float(sheets_raw.get("backoff_seconds", 1.5)),
    )
    profile = load_camera_profile(args.camera_profile)
    pipeline = RailVideoPipeline(
        settings=settings,
        camera_profile=profile,
        output_root=args.output_root,
        sheets_config=sheets_config,
    )

    if args.source:
        result = pipeline.process_video(
            video_path=args.source,
            run_name=args.run_name,
            delete_on_success=args.delete_on_success,
            export_excel=not args.no_excel,
            show_live=args.show_live,
        )
        summary = summarize_results([result])
    else:
        videos = discover_videos(args.source_dir)
        results = pipeline.process_batch(
            video_paths=videos,
            run_prefix=args.run_name,
            delete_on_success=args.delete_on_success,
            show_live=args.show_live,
        )
        summary = summarize_results(results)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
