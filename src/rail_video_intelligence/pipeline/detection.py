from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ultralytics import YOLO

from .types import DetectionRecord, PipelineSettings


LOGGER = logging.getLogger(__name__)


def _normalize_names(values: Sequence[str]) -> set[str]:
    return {v.strip().lower() for v in values if v and v.strip()}


def _resolve_target_class_ids(model: YOLO, settings: PipelineSettings) -> List[int]:
    names_map = model.names
    target_names = (
        _normalize_names(settings.train_class_names)
        | _normalize_names(settings.locomotive_class_names)
        | _normalize_names(settings.carriage_class_names)
    )
    target_ids = [class_id for class_id, class_name in names_map.items() if class_name.lower() in target_names]
    return sorted(set(target_ids))


def _label_role(class_name: str, settings: PipelineSettings) -> str:
    key = class_name.lower()
    if key in _normalize_names(settings.locomotive_class_names):
        return "locomotive"
    if key in _normalize_names(settings.carriage_class_names):
        return "carriage"
    if key in _normalize_names(settings.train_class_names):
        return "train"
    return class_name


def run_video_detection(
    video_path: Path,
    settings: PipelineSettings,
    output_root: Path,
    run_name: str,
    save_preview: bool = True,
    show_live: bool = False,
) -> tuple[List[DetectionRecord], Optional[Path], Dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(settings.model_path)
    target_ids = _resolve_target_class_ids(model, settings)

    run_kwargs = {
        "source": str(video_path),
        "conf": settings.confidence,
        "save": save_preview,
        "show": show_live,
        "stream": True,
        "project": str(output_root),
        "name": run_name,
        "exist_ok": True,
        "verbose": False,
    }
    if settings.device:
        run_kwargs["device"] = settings.device
    if target_ids:
        run_kwargs["classes"] = target_ids

    mode_used = settings.mode
    try:
        if settings.mode == "track":
            stream = model.track(**run_kwargs, tracker=settings.tracker, persist=True)
        else:
            stream = model.predict(**run_kwargs)
    except ModuleNotFoundError as exc:
        LOGGER.warning("Tracker dependency missing (%s), falling back to detect mode.", exc)
        mode_used = "detect"
        stream = model.predict(**run_kwargs)

    detections: List[DetectionRecord] = []
    frames_processed = 0
    frames_with_detections = 0
    class_histogram: Dict[str, int] = {}
    save_dir: Optional[Path] = None

    for frame_idx, result in enumerate(stream):
        frames_processed += 1
        if hasattr(result, "save_dir") and result.save_dir:
            save_dir = Path(str(result.save_dir))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        frames_with_detections += 1
        class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        confs = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        xyxy = boxes.xyxy.cpu().tolist()
        track_ids: List[Optional[int]]
        if boxes.id is None:
            track_ids = [None] * len(xyxy)
        else:
            track_ids = boxes.id.int().cpu().tolist()
        for idx, bbox in enumerate(xyxy):
            class_id = class_ids[idx] if idx < len(class_ids) else -1
            class_name = model.names.get(class_id, f"class_{class_id}")
            role = _label_role(class_name, settings)
            x1, y1, x2, y2 = (float(v) for v in bbox)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            confidence = float(confs[idx]) if idx < len(confs) else 0.0
            detections.append(
                DetectionRecord(
                    frame_index=frame_idx,
                    time_s=0.0,
                    class_id=class_id,
                    class_name=role,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=(cx, cy),
                    track_id=track_ids[idx],
                )
            )
            class_histogram[role] = class_histogram.get(role, 0) + 1

    preview_candidates = [
        output_root / run_name / video_path.name,
        output_root / run_name / f"{video_path.stem}.mp4",
    ]
    if save_dir:
        preview_candidates.extend(
            [
                save_dir / video_path.name,
                save_dir / f"{video_path.stem}.mp4",
            ]
        )
    preview_path = next((path for path in preview_candidates if path.exists()), None)

    stats: Dict[str, object] = {
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "mode_used": mode_used,
        "class_histogram": class_histogram,
    }
    return detections, preview_path, stats


def attach_detection_times(detections: List[DetectionRecord], fps: float) -> List[DetectionRecord]:
    if fps <= 0:
        fps = 30.0
    for det in detections:
        det.time_s = det.frame_index / fps
    return detections


def create_pseudo_track_ids_for_detect_mode(detections: List[DetectionRecord], max_dist_px: float = 80.0) -> None:
    """
    Lightweight tracker used only when model does not provide IDs (detect mode).
    """
    next_track = 1
    active: Dict[int, Tuple[int, float, float]] = {}
    detections.sort(key=lambda d: d.frame_index)
    for det in detections:
        if det.track_id is not None:
            continue
        best_id: Optional[int] = None
        best_dist = float("inf")
        for track_id, (last_frame, last_x, last_y) in active.items():
            if det.frame_index - last_frame > 3:
                continue
            dist = ((det.center[0] - last_x) ** 2 + (det.center[1] - last_y) ** 2) ** 0.5
            if dist < best_dist and dist <= max_dist_px:
                best_dist = dist
                best_id = track_id
        if best_id is None:
            best_id = next_track
            next_track += 1
        det.track_id = best_id
        active[best_id] = (det.frame_index, det.center[0], det.center[1])
