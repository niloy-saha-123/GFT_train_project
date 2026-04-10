from pathlib import Path

from rail_video_intelligence.pipeline.aggregation import aggregate_events
from rail_video_intelligence.pipeline.types import (
    CameraProfile,
    DetectionRecord,
    LaneConfig,
    MarkerConfig,
    PipelineSettings,
    VideoMetadata,
)


def _meta() -> VideoMetadata:
    return VideoMetadata(
        video_id="v1",
        source_path=Path("video.mov"),
        file_hash="abc",
        camera_id="camera_01",
        fps=30.0,
        width=1280,
        height=720,
        frame_count=300,
        duration_s=10.0,
    )


def _profile() -> CameraProfile:
    lane = LaneConfig(
        lane_id="lane_far",
        polygon=[(0, 0), (1279, 0), (1279, 719), (0, 719)],
        axis_start=(1270.0, 360.0),
        axis_end=(10.0, 360.0),
        direction_positive="left_to_right",
        direction_negative="right_to_left",
        marker=MarkerConfig(
            line_a=((900.0, 0.0), (900.0, 719.0)),
            line_b=((600.0, 0.0), (600.0, 719.0)),
            distance_m=60.0,
        ),
    )
    return CameraProfile(camera_id="camera_01", roi_polygon=lane.polygon, lanes=[lane], ignore_rectangles=[])


def _settings() -> PipelineSettings:
    return PipelineSettings(
        model_path="yolo.pt",
        tracker="bytetrack.yaml",
        mode="track",
        device=None,
        show_live=False,
        save_demo_overlay=False,
        confidence=0.25,
        train_class_names=["train"],
        locomotive_class_names=["locomotive"],
        carriage_class_names=["carriage"],
        event_gap_frames=5,
        merge_gap_frames=2,
        min_event_frames=2,
    )


def test_split_overlapping_opposite_direction_events():
    detections = [
        DetectionRecord(10, 10 / 30.0, 1, "train", 0.9, (980, 300, 1020, 350), (1000, 325), 1),
        DetectionRecord(11, 11 / 30.0, 1, "train", 0.9, (940, 300, 980, 350), (960, 325), 1),
        DetectionRecord(12, 12 / 30.0, 1, "train", 0.9, (900, 300, 940, 350), (920, 325), 1),
        DetectionRecord(13, 13 / 30.0, 1, "train", 0.9, (860, 300, 900, 350), (880, 325), 1),
        DetectionRecord(10, 10 / 30.0, 1, "train", 0.88, (520, 380, 560, 430), (540, 405), 2),
        DetectionRecord(11, 11 / 30.0, 1, "train", 0.88, (560, 380, 600, 430), (580, 405), 2),
        DetectionRecord(12, 12 / 30.0, 1, "train", 0.88, (600, 380, 640, 430), (620, 405), 2),
        DetectionRecord(13, 13 / 30.0, 1, "train", 0.88, (640, 380, 680, 430), (660, 405), 2),
    ]
    events = aggregate_events(_meta(), detections, _profile(), _settings())
    assert len(events) == 2
    directions = {evt.direction for evt in events}
    assert "right_to_left" in directions
    assert "left_to_right" in directions


def test_no_train_row_written():
    events = aggregate_events(_meta(), [], _profile(), _settings())
    assert len(events) == 1
    assert events[0].no_train_flag is True
    assert events[0].train_detected is False
