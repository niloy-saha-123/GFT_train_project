from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass(slots=True)
class VideoMetadata:
    video_id: str
    source_path: Path
    file_hash: str
    camera_id: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float


@dataclass(slots=True)
class DetectionRecord:
    frame_index: int
    time_s: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: BBox
    center: Point
    track_id: Optional[int]
    lane_id: Optional[str] = None


@dataclass(slots=True)
class MarkerConfig:
    line_a: Tuple[Point, Point]
    line_b: Tuple[Point, Point]
    distance_m: float


@dataclass(slots=True)
class LaneConfig:
    lane_id: str
    polygon: List[Point]
    axis_start: Point
    axis_end: Point
    direction_positive: str
    direction_negative: str
    marker: MarkerConfig


@dataclass(slots=True)
class CameraProfile:
    camera_id: str
    roi_polygon: List[Point]
    ignore_rectangles: List[Tuple[int, int, int, int]] = field(default_factory=list)
    lanes: List[LaneConfig] = field(default_factory=list)


@dataclass(slots=True)
class PipelineSettings:
    model_path: str
    tracker: str
    mode: str
    device: Optional[str]
    show_live: bool
    save_demo_overlay: bool
    confidence: float
    train_class_names: List[str]
    locomotive_class_names: List[str]
    carriage_class_names: List[str]
    event_gap_frames: int
    merge_gap_frames: int
    min_event_frames: int


@dataclass(slots=True)
class EventResult:
    video_id: str
    train_index: int
    camera_id: str
    track_id: str
    train_detected: bool
    no_train_flag: bool
    start_time_s: Optional[float]
    end_time_s: Optional[float]
    duration_s: Optional[float]
    direction: str
    speed_mph: Optional[float]
    speed_confidence: Optional[float]
    locomotive_count: Optional[int]
    carriage_count: Optional[int]
    company_name: str
    company_confidence: Optional[float]
    detection_confidence: float
    quality_flag: str
    processed_at_utc: str
    source_video: str
    notes: Optional[str] = None

    def as_sheet_row(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "train_index": self.train_index,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "train_detected": self.train_detected,
            "no_train_flag": self.no_train_flag,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "direction": self.direction,
            "speed_mph": self.speed_mph,
            "speed_confidence": self.speed_confidence,
            "locomotive_count": self.locomotive_count,
            "carriage_count": self.carriage_count,
            "company_name": self.company_name,
            "company_confidence": self.company_confidence,
            "detection_confidence": self.detection_confidence,
            "quality_flag": self.quality_flag,
            "processed_at_utc": self.processed_at_utc,
            "source_video": self.source_video,
            "notes": self.notes,
        }


@dataclass(slots=True)
class ProcessResult:
    metadata: VideoMetadata
    events: List[EventResult]
    output_json_path: Path
    preview_video_path: Optional[Path]
    demo_overlay_path: Optional[Path]
    sheet_sync_success: bool
    deleted_source: bool
