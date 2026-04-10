from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .aggregation import aggregate_events
from .company import enrich_company_labels
from .detection import (
    attach_detection_times,
    create_pseudo_track_ids_for_detect_mode,
    run_video_detection,
)
from .ingest import extract_video_metadata
from .sheets import GoogleSheetsWriter, SheetsConfig, write_rows_to_csv, write_rows_to_excel
from .types import CameraProfile, EventResult, PipelineSettings, ProcessResult
from .visualize import render_demo_overlay


LOGGER = logging.getLogger(__name__)


class RailVideoPipeline:
    def __init__(
        self,
        settings: PipelineSettings,
        camera_profile: CameraProfile,
        output_root: Path,
        sheets_config: Optional[SheetsConfig] = None,
    ) -> None:
        self.settings = settings
        self.camera_profile = camera_profile
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sheet_writer = GoogleSheetsWriter(sheets_config or SheetsConfig(False, None, "events", None))

    def process_video(
        self,
        video_path: Path,
        run_name: str,
        delete_on_success: bool = False,
        export_excel: bool = True,
        show_live: Optional[bool] = None,
    ) -> ProcessResult:
        effective_show_live = self.settings.show_live if show_live is None else show_live
        metadata = extract_video_metadata(video_path)
        detections, preview_path, det_stats = run_video_detection(
            video_path=video_path,
            settings=self.settings,
            output_root=self.output_root,
            run_name=run_name,
            save_preview=True,
            show_live=effective_show_live,
        )
        detections = attach_detection_times(detections, metadata.fps)
        if det_stats.get("mode_used") == "detect":
            create_pseudo_track_ids_for_detect_mode(detections)

        events = aggregate_events(metadata=metadata, detections=detections, profile=self.camera_profile, settings=self.settings)
        events = enrich_company_labels(metadata=metadata, events=events, detections=detections)
        rows = [evt.as_sheet_row() for evt in events]

        run_dir = self.output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        demo_overlay_path: Optional[Path] = None
        if self.settings.save_demo_overlay:
            demo_overlay_path = render_demo_overlay(
                video_path=video_path,
                output_path=run_dir / "demo_overlay.mp4",
                profile=self.camera_profile,
                detections=detections,
                events=events,
                fps_hint=metadata.fps,
            )

        event_json = run_dir / "event_summary.json"
        payload = {
            "metadata": asdict(metadata),
            "detection_stats": det_stats,
            "preview_video_path": str(preview_path) if preview_path else None,
            "demo_overlay_path": str(demo_overlay_path) if demo_overlay_path else None,
            "events": rows,
        }
        with event_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

        csv_path = run_dir / "event_rows.csv"
        write_rows_to_csv(rows, csv_path)
        if export_excel:
            write_rows_to_excel(rows, run_dir / "event_rows.xlsx")

        sheets_ok = self.sheet_writer.upsert_rows(rows, key_fields=("video_id", "train_index"))

        deleted_source = False
        if delete_on_success and sheets_ok:
            try:
                video_path.unlink(missing_ok=False)
                deleted_source = True
            except FileNotFoundError:
                deleted_source = False
            except Exception as exc:
                LOGGER.error("Video delete failed after success sync: %s", exc)
                deleted_source = False

        return ProcessResult(
            metadata=metadata,
            events=events,
            output_json_path=event_json,
            preview_video_path=demo_overlay_path or preview_path,
            demo_overlay_path=demo_overlay_path,
            sheet_sync_success=sheets_ok,
            deleted_source=deleted_source,
        )

    def process_batch(
        self,
        video_paths: List[Path],
        run_prefix: str,
        delete_on_success: bool = False,
        show_live: Optional[bool] = None,
    ) -> List[ProcessResult]:
        results: List[ProcessResult] = []
        for idx, path in enumerate(video_paths, start=1):
            run_name = f"{run_prefix}_{idx:03d}_{path.stem}"
            results.append(
                self.process_video(
                    video_path=path,
                    run_name=run_name,
                    delete_on_success=delete_on_success,
                    show_live=show_live,
                )
            )
        return results


def summarize_results(results: List[ProcessResult]) -> Dict[str, object]:
    total_events = sum(len(r.events) for r in results)
    no_train_videos = sum(1 for r in results if r.events and r.events[0].no_train_flag)
    deleted_files = sum(1 for r in results if r.deleted_source)
    sheet_success = sum(1 for r in results if r.sheet_sync_success)
    return {
        "videos_processed": len(results),
        "total_events": total_events,
        "no_train_videos": no_train_videos,
        "sheet_sync_success_count": sheet_success,
        "deleted_source_files": deleted_files,
    }
