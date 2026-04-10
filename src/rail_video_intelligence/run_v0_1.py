#!/usr/bin/env python3
"""Baseline v0.1 train pass detector + tracker + optional tripwire speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import yaml
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v0.1 baseline rail video pipeline.")
    parser.add_argument("--config", type=Path, help="YAML config file with defaults.")
    parser.add_argument("--source", type=Path, help="Input video path.")
    parser.add_argument("--project", type=Path, default=Path("outputs/v0_1"), help="Output root directory.")
    parser.add_argument("--name", default="baseline", help="Run name under output root.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path or name.")
    parser.add_argument(
        "--mode",
        choices=("track", "detect"),
        default="track",
        help="track: detection+tracking, detect: detection only (no tracker dependency).",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config for Ultralytics.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--train-class", type=int, default=6, help="COCO train class id.")
    parser.add_argument("--camera-id", default="camera_01", help="Camera identifier in output summary.")
    parser.add_argument("--track-id", default="track_01", help="Track identifier in output summary.")
    parser.add_argument("--tripwire-a", type=float, help="Normalized x (0-1) for first virtual line.")
    parser.add_argument("--tripwire-b", type=float, help="Normalized x (0-1) for second virtual line.")
    parser.add_argument("--distance-m", type=float, help="Real-world distance between tripwires in meters.")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live inference window.",
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated output video.",
    )
    return parser


def merge_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not args.config:
        return args

    with args.config.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain key/value map: {args.config}")

    for key, value in loaded.items():
        normalized = key.replace("-", "_")
        if not hasattr(args, normalized):
            continue
        if getattr(args, normalized) == parser.get_default(normalized):
            setattr(args, normalized, value)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.source is None:
        parser.error("--source required (or set source in --config).")

    wire_args = [args.tripwire_a, args.tripwire_b, args.distance_m]
    if any(value is not None for value in wire_args) and not all(value is not None for value in wire_args):
        parser.error("For speed, set all: --tripwire-a --tripwire-b --distance-m.")

    if args.tripwire_a is not None:
        if not (0.0 <= args.tripwire_a <= 1.0 and 0.0 <= args.tripwire_b <= 1.0):
            parser.error("Tripwire values must be normalized between 0 and 1.")
        if args.tripwire_a == args.tripwire_b:
            parser.error("Tripwire A and B cannot be equal.")
        if args.distance_m <= 0:
            parser.error("--distance-m must be positive.")


def read_video_meta(source: Path) -> Tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if fps <= 0:
        fps = 30.0
    return fps, width, height, frame_count


def crossing_frame(points: List[Tuple[int, float, float]], wire_x: float) -> Optional[float]:
    if len(points) < 2:
        return None
    for (f1, x1, _), (f2, x2, _) in zip(points, points[1:]):
        if x1 == wire_x:
            return float(f1)
        if (x1 - wire_x) * (x2 - wire_x) <= 0:
            if x2 == x1:
                return float(f2)
            ratio = (wire_x - x1) / (x2 - x1)
            return float(f1) + ratio * float(f2 - f1)
    return None


def estimate_direction(points: List[Tuple[int, float, float]]) -> str:
    if len(points) < 2:
        return "unknown"
    dx = points[-1][1] - points[0][1]
    if abs(dx) < 2.0:
        return "unknown"
    return "left_to_right" if dx > 0 else "right_to_left"


def track_points(
    result_stream: Iterable,
    class_id: int,
    allow_untracked: bool = False,
) -> Tuple[Dict[int, List[Tuple[int, float, float]]], int, int, Optional[int], Optional[int]]:
    history: Dict[int, List[Tuple[int, float, float]]] = {}
    total_frames = 0
    frames_with_detection = 0
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    for frame_index, result in enumerate(result_stream):
        total_frames += 1
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        frames_with_detection += 1
        if start_frame is None:
            start_frame = frame_index
        end_frame = frame_index

        if boxes.id is None or boxes.cls is None:
            if not allow_untracked or boxes.cls is None or boxes.conf is None:
                continue
            classes = boxes.cls.int().cpu().tolist()
            xywh = boxes.xywh.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            candidates = [
                (idx, conf)
                for idx, (det_cls, conf) in enumerate(zip(classes, confs))
                if det_cls == class_id
            ]
            if not candidates:
                continue
            best_idx = max(candidates, key=lambda row: row[1])[0]
            x, y, _, _ = xywh[best_idx]
            history.setdefault(0, []).append((frame_index, float(x), float(y)))
            continue

        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        xywh = boxes.xywh.cpu().tolist()

        for det_id, det_cls, (x, y, _, _) in zip(ids, classes, xywh):
            if det_cls != class_id:
                continue
            history.setdefault(det_id, []).append((frame_index, float(x), float(y)))

    return history, total_frames, frames_with_detection, start_frame, end_frame


def build_summary(
    args: argparse.Namespace,
    fps: float,
    width: int,
    frame_count_meta: int,
    history: Dict[int, List[Tuple[int, float, float]]],
    total_frames_processed: int,
    frames_with_detection: int,
    start_frame: Optional[int],
    end_frame: Optional[int],
    run_dir: Path,
) -> Dict[str, object]:
    primary_track_id: Optional[int] = None
    primary_points: List[Tuple[int, float, float]] = []
    if history:
        primary_track_id, primary_points = max(history.items(), key=lambda item: len(item[1]))

    train_detected = start_frame is not None
    start_time_s = round(start_frame / fps, 3) if start_frame is not None else None
    end_time_s = round(end_frame / fps, 3) if end_frame is not None else None
    duration_s = round(end_time_s - start_time_s, 3) if train_detected else None
    direction = estimate_direction(primary_points) if primary_points else "unknown"

    speed_mph = None
    speed_confidence = None
    if (
        primary_points
        and args.tripwire_a is not None
        and args.tripwire_b is not None
        and args.distance_m is not None
        and width > 0
    ):
        wire_a_px = args.tripwire_a * width
        wire_b_px = args.tripwire_b * width
        frame_a = crossing_frame(primary_points, wire_a_px)
        frame_b = crossing_frame(primary_points, wire_b_px)
        if frame_a is not None and frame_b is not None and frame_a != frame_b:
            dt = abs(frame_b - frame_a) / fps
            speed_mps = args.distance_m / dt if dt > 0 else 0.0
            speed_mph = round(speed_mps * 2.23694, 3)
            points_score = min(len(primary_points) / 30.0, 1.0)
            speed_confidence = round(points_score, 3)

    coverage = frames_with_detection / max(total_frames_processed, 1)
    points_score = min(len(primary_points) / 30.0, 1.0) if primary_points else 0.0
    detection_confidence = round((coverage + points_score) / 2.0, 3)

    output_video_path = run_dir / args.source.name if args.save_video else None
    if output_video_path and not output_video_path.exists():
        output_video_path = None

    return {
        "video_id": args.source.stem,
        "train_detected": train_detected,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": duration_s,
        "direction": direction,
        "direction_cardinal": "unknown",
        "speed_mph": speed_mph,
        "speed_confidence": speed_confidence,
        "locomotive_count": None,
        "consist_count": None,
        "train_type": "unknown",
        "train_type_confidence": None,
        "camera_id": args.camera_id,
        "track_id": args.track_id,
        "processing_version": "v0.1.0",
        "model": args.model,
        "mode": args.mode,
        "tracker": args.tracker,
        "detection_confidence": detection_confidence,
        "total_frames_processed": total_frames_processed,
        "total_frames_video_meta": frame_count_meta,
        "frames_with_detection": frames_with_detection,
        "primary_track_id": primary_track_id,
        "primary_track_points": len(primary_points),
        "output_video_path": str(output_video_path) if output_video_path else None,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = merge_config(args, parser)

    if isinstance(args.source, str):
        args.source = Path(args.source)
    if isinstance(args.project, str):
        args.project = Path(args.project)
    if isinstance(args.config, str):
        args.config = Path(args.config)

    validate_args(args, parser)

    fps, width, _height, frame_count_meta = read_video_meta(args.source)
    run_dir = args.project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    if args.mode == "track":
        results = model.track(
            source=str(args.source),
            conf=args.conf,
            classes=[args.train_class],
            tracker=args.tracker,
            save=args.save_video,
            show=args.show,
            stream=True,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            verbose=False,
            persist=True,
        )
    else:
        results = model.predict(
            source=str(args.source),
            conf=args.conf,
            classes=[args.train_class],
            save=args.save_video,
            show=args.show,
            stream=True,
            project=str(args.project),
            name=args.name,
            exist_ok=True,
            verbose=False,
        )

    history, total_frames_processed, frames_with_detection, start_frame, end_frame = track_points(
        result_stream=results,
        class_id=args.train_class,
        allow_untracked=args.mode == "detect",
    )

    summary = build_summary(
        args=args,
        fps=fps,
        width=width,
        frame_count_meta=frame_count_meta,
        history=history,
        total_frames_processed=total_frames_processed,
        frames_with_detection=frames_with_detection,
        start_frame=start_frame,
        end_frame=end_frame,
        run_dir=run_dir,
    )

    summary_path = run_dir / "event_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Summary saved: {summary_path}")
    if summary["output_video_path"]:
        print(f"Annotated video: {summary['output_video_path']}")


if __name__ == "__main__":
    main()
