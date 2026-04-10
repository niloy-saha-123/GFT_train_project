from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .types import CameraProfile, DetectionRecord, EventResult


_PALETTE = [
    (255, 99, 71),
    (0, 191, 255),
    (124, 252, 0),
    (255, 215, 0),
    (186, 85, 211),
    (64, 224, 208),
]


def _to_int_point(point: Tuple[float, float]) -> Tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _lane_color(lane_id: str) -> Tuple[int, int, int]:
    index = abs(hash(lane_id)) % len(_PALETTE)
    return _PALETTE[index]


def _class_color(name: str) -> Tuple[int, int, int]:
    if name == "train":
        return (0, 0, 255)
    if name == "locomotive":
        return (255, 0, 0)
    if name == "carriage":
        return (0, 255, 255)
    return (255, 255, 255)


def _event_bounds(events: List[EventResult], fps: float) -> List[Tuple[int, int, EventResult]]:
    bounds = []
    for evt in events:
        if evt.no_train_flag or evt.start_time_s is None or evt.end_time_s is None:
            continue
        start = max(0, int(round(evt.start_time_s * fps)))
        end = max(start, int(round(evt.end_time_s * fps)))
        bounds.append((start, end, evt))
    return bounds


def render_demo_overlay(
    video_path: Path,
    output_path: Path,
    profile: CameraProfile,
    detections: List[DetectionRecord],
    events: List[EventResult],
    fps_hint: float,
) -> Optional[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = fps_hint if fps_hint > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        return None

    detections_by_frame: Dict[int, List[DetectionRecord]] = defaultdict(list)
    for det in detections:
        detections_by_frame[det.frame_index].append(det)
    bounds = _event_bounds(events, fps)

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if profile.roi_polygon:
                roi_pts = np.array([_to_int_point(pt) for pt in profile.roi_polygon], dtype=np.int32)
                cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, "ROI", _to_int_point(profile.roi_polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for lane in profile.lanes:
                color = _lane_color(lane.lane_id)
                lane_pts = np.array([_to_int_point(pt) for pt in lane.polygon], dtype=np.int32)
                cv2.polylines(frame, [lane_pts], isClosed=True, color=color, thickness=2)
                cv2.arrowedLine(frame, _to_int_point(lane.axis_start), _to_int_point(lane.axis_end), color, 2, tipLength=0.03)
                cv2.line(frame, _to_int_point(lane.marker.line_a[0]), _to_int_point(lane.marker.line_a[1]), (255, 255, 0), 2)
                cv2.line(frame, _to_int_point(lane.marker.line_b[0]), _to_int_point(lane.marker.line_b[1]), (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"{lane.lane_id} A",
                    _to_int_point(lane.marker.line_a[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{lane.lane_id} B",
                    _to_int_point(lane.marker.line_b[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

            for det in detections_by_frame.get(frame_idx, []):
                x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
                color = _class_color(det.class_name)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, _to_int_point(det.center), radius=3, color=color, thickness=-1)
                label = f"{det.class_name}|id={det.track_id}|{det.lane_id or 'no_lane'}"
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            active_events = [evt for start, end, evt in bounds if start <= frame_idx <= end]
            for idx, evt in enumerate(active_events):
                cv2.putText(
                    frame,
                    f"E{evt.train_index} {evt.track_id} {evt.direction} {evt.speed_mph or 0:.1f} mph",
                    (10, 24 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            writer.write(frame)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    return output_path if output_path.exists() else None
