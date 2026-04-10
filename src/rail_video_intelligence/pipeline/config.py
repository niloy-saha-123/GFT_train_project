from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .types import CameraProfile, LaneConfig, MarkerConfig, PipelineSettings


def _to_point_list(points: List[List[float]]) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def load_camera_profile(path: Path) -> CameraProfile:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    lanes: List[LaneConfig] = []
    for lane in raw.get("lanes", []):
        marker_raw = lane["marker"]
        marker = MarkerConfig(
            line_a=(
                (float(marker_raw["line_a"][0][0]), float(marker_raw["line_a"][0][1])),
                (float(marker_raw["line_a"][1][0]), float(marker_raw["line_a"][1][1])),
            ),
            line_b=(
                (float(marker_raw["line_b"][0][0]), float(marker_raw["line_b"][0][1])),
                (float(marker_raw["line_b"][1][0]), float(marker_raw["line_b"][1][1])),
            ),
            distance_m=float(marker_raw["distance_m"]),
        )
        lanes.append(
            LaneConfig(
                lane_id=str(lane["lane_id"]),
                polygon=_to_point_list(lane["polygon"]),
                axis_start=(float(lane["axis_start"][0]), float(lane["axis_start"][1])),
                axis_end=(float(lane["axis_end"][0]), float(lane["axis_end"][1])),
                direction_positive=str(lane["direction_positive"]),
                direction_negative=str(lane["direction_negative"]),
                marker=marker,
            )
        )

    return CameraProfile(
        camera_id=str(raw["camera_id"]),
        roi_polygon=_to_point_list(raw.get("roi_polygon", [])),
        ignore_rectangles=[
            (int(x), int(y), int(w), int(h))
            for x, y, w, h in raw.get("ignore_rectangles", [])
        ],
        lanes=lanes,
    )


def load_pipeline_settings(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_pipeline_settings(raw: Dict[str, Any]) -> PipelineSettings:
    device_raw = raw.get("device")
    device = None
    if isinstance(device_raw, str) and device_raw.strip() and device_raw.strip().lower() != "auto":
        device = device_raw.strip()
    return PipelineSettings(
        model_path=str(raw.get("model_path", "yolov8n.pt")),
        tracker=str(raw.get("tracker", "bytetrack.yaml")),
        mode=str(raw.get("mode", "track")),
        device=device,
        show_live=bool(raw.get("show_live", False)),
        save_demo_overlay=bool(raw.get("save_demo_overlay", True)),
        confidence=float(raw.get("confidence", 0.25)),
        train_class_names=list(raw.get("train_class_names", ["train"])),
        locomotive_class_names=list(raw.get("locomotive_class_names", ["locomotive"])),
        carriage_class_names=list(raw.get("carriage_class_names", ["carriage", "railcar", "freight car"])),
        event_gap_frames=int(raw.get("event_gap_frames", 30)),
        merge_gap_frames=int(raw.get("merge_gap_frames", 20)),
        min_event_frames=int(raw.get("min_event_frames", 6)),
    )


def save_camera_profile(profile: CameraProfile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "camera_id": profile.camera_id,
        "roi_polygon": [[x, y] for x, y in profile.roi_polygon],
        "ignore_rectangles": [list(item) for item in profile.ignore_rectangles],
        "lanes": [],
    }
    for lane in profile.lanes:
        payload["lanes"].append(
            {
                "lane_id": lane.lane_id,
                "polygon": [[x, y] for x, y in lane.polygon],
                "axis_start": [lane.axis_start[0], lane.axis_start[1]],
                "axis_end": [lane.axis_end[0], lane.axis_end[1]],
                "direction_positive": lane.direction_positive,
                "direction_negative": lane.direction_negative,
                "marker": {
                    "line_a": [[lane.marker.line_a[0][0], lane.marker.line_a[0][1]], [lane.marker.line_a[1][0], lane.marker.line_a[1][1]]],
                    "line_b": [[lane.marker.line_b[0][0], lane.marker.line_b[0][1]], [lane.marker.line_b[1][0], lane.marker.line_b[1][1]]],
                    "distance_m": lane.marker.distance_m,
                },
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
