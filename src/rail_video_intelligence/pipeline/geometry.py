from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1):
            inside = not inside
    return inside


def point_line_side(point: Point, line: Tuple[Point, Point]) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def line_crossing_time(
    points: List[Tuple[int, float, float]],
    line: Tuple[Point, Point],
) -> Optional[float]:
    if len(points) < 2:
        return None
    for (f1, x1, y1), (f2, x2, y2) in zip(points, points[1:]):
        s1 = point_line_side((x1, y1), line)
        s2 = point_line_side((x2, y2), line)
        if s1 == 0:
            return float(f1)
        if s1 * s2 <= 0:
            denom = (s1 - s2)
            if denom == 0:
                return float(f2)
            ratio = s1 / denom
            return float(f1) + ratio * float(f2 - f1)
    return None


def project_on_axis(point: Point, axis_start: Point, axis_end: Point) -> float:
    px, py = point
    ax, ay = axis_start
    bx, by = axis_end
    vx, vy = bx - ax, by - ay
    mag2 = vx * vx + vy * vy
    if mag2 == 0:
        return 0.0
    return ((px - ax) * vx + (py - ay) * vy) / mag2


def estimate_direction(
    points: Iterable[Point],
    axis_start: Point,
    axis_end: Point,
    positive_label: str,
    negative_label: str,
    min_delta: float = 0.01,
) -> str:
    points_list = list(points)
    if len(points_list) < 2:
        return "unknown"
    start = project_on_axis(points_list[0], axis_start, axis_end)
    end = project_on_axis(points_list[-1], axis_start, axis_end)
    delta = end - start
    if abs(delta) < min_delta:
        return "unknown"
    return positive_label if delta > 0 else negative_label


def speed_mph_from_crossings(
    frame_a: Optional[float],
    frame_b: Optional[float],
    fps: float,
    distance_m: float,
) -> Optional[float]:
    if frame_a is None or frame_b is None or frame_a == frame_b:
        return None
    if fps <= 0 or distance_m <= 0:
        return None
    dt = abs(frame_b - frame_a) / fps
    if dt <= 0:
        return None
    speed_mps = distance_m / dt
    return speed_mps * 2.23694
