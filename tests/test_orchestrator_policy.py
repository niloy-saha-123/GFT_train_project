from pathlib import Path

from rail_video_intelligence.pipeline.orchestrator import RailVideoPipeline
from rail_video_intelligence.pipeline.sheets import SheetsConfig
from rail_video_intelligence.pipeline.types import (
    CameraProfile,
    EventResult,
    LaneConfig,
    MarkerConfig,
    PipelineSettings,
    VideoMetadata,
)


def _settings() -> PipelineSettings:
    return PipelineSettings(
        model_path="yolov8n.pt",
        tracker="bytetrack.yaml",
        mode="detect",
        device=None,
        show_live=False,
        save_demo_overlay=False,
        confidence=0.25,
        train_class_names=["train"],
        locomotive_class_names=["locomotive"],
        carriage_class_names=["carriage"],
        event_gap_frames=30,
        merge_gap_frames=20,
        min_event_frames=3,
    )


def _profile() -> CameraProfile:
    lane = LaneConfig(
        lane_id="lane_1",
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        axis_start=(100.0, 50.0),
        axis_end=(0.0, 50.0),
        direction_positive="left_to_right",
        direction_negative="right_to_left",
        marker=MarkerConfig(line_a=((80, 0), (80, 100)), line_b=((20, 0), (20, 100)), distance_m=30.0),
    )
    return CameraProfile(camera_id="camera_01", roi_polygon=lane.polygon, ignore_rectangles=[], lanes=[lane])


def test_delete_on_success_only(monkeypatch, tmp_path: Path):
    from rail_video_intelligence.pipeline import orchestrator as mod

    video = tmp_path / "camera_01_2026_01_01_000000.MOV"
    video.write_bytes(b"fake")

    metadata = VideoMetadata(
        video_id="camera_01_2026_01_01_000000",
        source_path=video,
        file_hash="abc",
        camera_id="camera_01",
        fps=30.0,
        width=100,
        height=100,
        frame_count=300,
        duration_s=10.0,
    )
    event = EventResult(
        video_id=metadata.video_id,
        train_index=1,
        camera_id=metadata.camera_id,
        track_id="lane_1",
        train_detected=True,
        no_train_flag=False,
        start_time_s=1.0,
        end_time_s=3.0,
        duration_s=2.0,
        direction="right_to_left",
        speed_mph=40.0,
        speed_confidence=0.8,
        locomotive_count=1,
        carriage_count=4,
        company_name="Unknown",
        company_confidence=None,
        detection_confidence=0.9,
        quality_flag="ok",
        processed_at_utc="2026-01-01T00:00:00+00:00",
        source_video=str(video),
    )

    monkeypatch.setattr(mod, "extract_video_metadata", lambda _video_path: metadata)
    monkeypatch.setattr(mod, "run_video_detection", lambda **kwargs: ([], None, {"mode_used": "detect"}))
    monkeypatch.setattr(mod, "attach_detection_times", lambda detections, fps: detections)
    monkeypatch.setattr(mod, "create_pseudo_track_ids_for_detect_mode", lambda detections: None)
    monkeypatch.setattr(mod, "aggregate_events", lambda **kwargs: [event])
    monkeypatch.setattr(mod, "enrich_company_labels", lambda **kwargs: [event])

    pipeline = RailVideoPipeline(
        settings=_settings(),
        camera_profile=_profile(),
        output_root=tmp_path / "out",
        sheets_config=SheetsConfig(enabled=False, spreadsheet_id=None, worksheet_name="events", credentials_path=None),
    )

    monkeypatch.setattr(pipeline.sheet_writer, "upsert_rows", lambda rows, key_fields: False)
    result = pipeline.process_video(video, run_name="run1", delete_on_success=True)
    assert result.deleted_source is False
    assert video.exists()

    monkeypatch.setattr(pipeline.sheet_writer, "upsert_rows", lambda rows, key_fields: True)
    result2 = pipeline.process_video(video, run_name="run2", delete_on_success=True)
    assert result2.deleted_source is True
    assert not video.exists()
