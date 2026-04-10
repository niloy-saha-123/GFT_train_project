from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from .config import save_camera_profile
from .ingest import sample_frames
from .types import CameraProfile, LaneConfig, MarkerConfig


Point = Tuple[float, float]


@dataclass(slots=True)
class CalibrationSelection:
    roi_polygon: List[Point]
    ignore_rectangles: List[tuple[int, int, int, int]]
    lanes: List[LaneConfig]


def _pick_points(image: np.ndarray, title: str, n: int) -> List[Point]:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))
    plt.imshow(image[:, :, ::-1])
    plt.title(title)
    pts = plt.ginput(n=n, timeout=0)
    plt.close()
    return [(float(x), float(y)) for x, y in pts]


def interactive_calibration(video_path: Path, camera_id: str, frame_index: int = 0) -> CalibrationSelection:
    samples = sample_frames(video_path, [frame_index])
    if not samples:
        raise RuntimeError("No frame available for calibration.")
    _, frame = samples[0]

    roi = _pick_points(frame, "Select ROI polygon points, then close window", n=-1)
    ignore = _pick_points(frame, "Select 2 points (top-left, bottom-right) for timestamp mask", n=2)
    if len(ignore) == 2:
        x1, y1 = ignore[0]
        x2, y2 = ignore[1]
        ignore_rectangles = [(int(min(x1, x2)), int(min(y1, y2)), int(abs(x2 - x1)), int(abs(y2 - y1)))]
    else:
        ignore_rectangles = []

    lane_poly = _pick_points(frame, "Lane polygon points, then close window", n=-1)
    axis = _pick_points(frame, "Lane axis 2 points (start/end)", n=2)
    line_a = _pick_points(frame, "Speed marker A line: 2 points", n=2)
    line_b = _pick_points(frame, "Speed marker B line: 2 points", n=2)
    distance_m = float(input("Enter real-world distance (meters) between marker A and B: ").strip())
    direction_positive = input("Direction label for axis positive movement: ").strip() or "positive"
    direction_negative = input("Direction label for axis negative movement: ").strip() or "negative"

    lane = LaneConfig(
        lane_id="lane_1",
        polygon=lane_poly,
        axis_start=axis[0],
        axis_end=axis[1],
        direction_positive=direction_positive,
        direction_negative=direction_negative,
        marker=MarkerConfig(line_a=(line_a[0], line_a[1]), line_b=(line_b[0], line_b[1]), distance_m=distance_m),
    )
    return CalibrationSelection(roi_polygon=roi, ignore_rectangles=ignore_rectangles, lanes=[lane])


def save_interactive_profile(video_path: Path, output_profile_path: Path, camera_id: str, frame_index: int = 0) -> CameraProfile:
    selection = interactive_calibration(video_path=video_path, camera_id=camera_id, frame_index=frame_index)
    profile = CameraProfile(
        camera_id=camera_id,
        roi_polygon=selection.roi_polygon,
        ignore_rectangles=selection.ignore_rectangles,
        lanes=selection.lanes,
    )
    save_camera_profile(profile, output_profile_path)
    return profile
