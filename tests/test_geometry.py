from rail_video_intelligence.pipeline.geometry import (
    estimate_direction,
    line_crossing_time,
    speed_mph_from_crossings,
)


def test_line_crossing_time_interpolates():
    points = [(10, 0.0, 0.0), (20, 10.0, 0.0)]
    line = ((5.0, -10.0), (5.0, 10.0))
    crossing = line_crossing_time(points, line)
    assert crossing is not None
    assert abs(crossing - 15.0) < 1e-6


def test_speed_formula():
    speed = speed_mph_from_crossings(frame_a=100.0, frame_b=130.0, fps=30.0, distance_m=30.0)
    assert speed is not None
    assert round(speed, 3) == 67.108


def test_direction_mapping_axis_projection():
    points = [(10.0, 10.0), (20.0, 12.0), (40.0, 15.0)]
    direction = estimate_direction(
        points=points,
        axis_start=(0.0, 0.0),
        axis_end=(100.0, 0.0),
        positive_label="eastbound",
        negative_label="westbound",
    )
    assert direction == "eastbound"
