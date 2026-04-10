from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .geometry import (
    estimate_direction,
    line_crossing_time,
    point_in_polygon,
    speed_mph_from_crossings,
)
from .types import CameraProfile, DetectionRecord, EventResult, LaneConfig, PipelineSettings, VideoMetadata


@dataclass(slots=True)
class EventCandidate:
    lane_id: str
    track_id: int
    start_frame: int
    end_frame: int
    detections: List[DetectionRecord]
    direction: str
    speed_mph: Optional[float]
    speed_confidence: Optional[float]


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def assign_lane_ids(detections: Iterable[DetectionRecord], profile: CameraProfile) -> None:
    for det in detections:
        lane_id = None
        for lane in profile.lanes:
            if point_in_polygon(det.center, lane.polygon):
                lane_id = lane.lane_id
                break
        det.lane_id = lane_id


def _split_by_gap(records: List[DetectionRecord], gap_frames: int) -> List[List[DetectionRecord]]:
    if not records:
        return []
    chunks: List[List[DetectionRecord]] = []
    current = [records[0]]
    for prev, cur in zip(records, records[1:]):
        if cur.frame_index - prev.frame_index > gap_frames:
            chunks.append(current)
            current = [cur]
        else:
            current.append(cur)
    chunks.append(current)
    return chunks


def _lane_lookup(profile: CameraProfile) -> Dict[str, LaneConfig]:
    return {lane.lane_id: lane for lane in profile.lanes}


def _build_train_candidates(
    detections: List[DetectionRecord],
    profile: CameraProfile,
    settings: PipelineSettings,
    fps: float,
) -> List[EventCandidate]:
    train_detections = [d for d in detections if d.class_name == "train" and d.lane_id and d.track_id is not None]
    grouped: Dict[Tuple[str, int], List[DetectionRecord]] = defaultdict(list)
    for det in train_detections:
        grouped[(det.lane_id or "unknown", int(det.track_id))].append(det)

    lanes = _lane_lookup(profile)
    candidates: List[EventCandidate] = []
    for (lane_id, track_id), records in grouped.items():
        lane = lanes.get(lane_id)
        if lane is None:
            continue
        records.sort(key=lambda r: r.frame_index)
        for chunk in _split_by_gap(records, settings.event_gap_frames):
            if len(chunk) < settings.min_event_frames:
                continue
            path = [d.center for d in chunk]
            direction = estimate_direction(
                points=path,
                axis_start=lane.axis_start,
                axis_end=lane.axis_end,
                positive_label=lane.direction_positive,
                negative_label=lane.direction_negative,
            )
            points = [(d.frame_index, d.center[0], d.center[1]) for d in chunk]
            cross_a = line_crossing_time(points, lane.marker.line_a)
            cross_b = line_crossing_time(points, lane.marker.line_b)
            speed_mph = speed_mph_from_crossings(cross_a, cross_b, fps=fps, distance_m=lane.marker.distance_m)
            speed_confidence = None
            if speed_mph is not None:
                speed_confidence = round(min(len(chunk) / 40.0, 1.0), 3)
            candidates.append(
                EventCandidate(
                    lane_id=lane_id,
                    track_id=track_id,
                    start_frame=chunk[0].frame_index,
                    end_frame=chunk[-1].frame_index,
                    detections=chunk,
                    direction=direction,
                    speed_mph=round(speed_mph, 3) if speed_mph is not None else None,
                    speed_confidence=speed_confidence,
                )
            )
    candidates.sort(key=lambda c: c.start_frame)
    return candidates


def _merge_candidates(candidates: List[EventCandidate], merge_gap_frames: int) -> List[EventCandidate]:
    if not candidates:
        return []
    merged: List[EventCandidate] = []
    for cand in candidates:
        if not merged:
            merged.append(cand)
            continue
        prev = merged[-1]
        same_lane = prev.lane_id == cand.lane_id
        same_track = prev.track_id == cand.track_id
        same_direction = prev.direction == cand.direction
        if same_lane and same_track and same_direction and cand.start_frame - prev.end_frame <= merge_gap_frames:
            prev.end_frame = cand.end_frame
            prev.detections.extend(cand.detections)
            if prev.speed_mph is None:
                prev.speed_mph = cand.speed_mph
                prev.speed_confidence = cand.speed_confidence
        else:
            merged.append(cand)
    return merged


def _count_crossing_ids(
    detections: List[DetectionRecord],
    lane: LaneConfig,
    start_frame: int,
    end_frame: int,
    class_name: str,
) -> int:
    filtered = [
        d for d in detections
        if d.class_name == class_name and d.lane_id == lane.lane_id and d.track_id is not None and start_frame <= d.frame_index <= end_frame
    ]
    grouped: Dict[int, List[DetectionRecord]] = defaultdict(list)
    for det in filtered:
        grouped[int(det.track_id)].append(det)
    count = 0
    for records in grouped.values():
        records.sort(key=lambda r: r.frame_index)
        points = [(r.frame_index, r.center[0], r.center[1]) for r in records]
        crossed = line_crossing_time(points, lane.marker.line_a) is not None
        if crossed:
            count += 1
    return count


def aggregate_events(
    metadata: VideoMetadata,
    detections: List[DetectionRecord],
    profile: CameraProfile,
    settings: PipelineSettings,
) -> List[EventResult]:
    assign_lane_ids(detections, profile)
    candidates = _build_train_candidates(detections=detections, profile=profile, settings=settings, fps=metadata.fps)
    candidates = _merge_candidates(candidates, merge_gap_frames=settings.merge_gap_frames)
    if not candidates:
        return [
            EventResult(
                video_id=metadata.video_id,
                train_index=0,
                camera_id=metadata.camera_id,
                track_id="none",
                train_detected=False,
                no_train_flag=True,
                start_time_s=None,
                end_time_s=None,
                duration_s=None,
                direction="unknown",
                speed_mph=None,
                speed_confidence=None,
                locomotive_count=0,
                carriage_count=0,
                company_name="Unknown",
                company_confidence=None,
                detection_confidence=0.0,
                quality_flag="no_train",
                processed_at_utc=_now_utc_iso(),
                source_video=str(metadata.source_path),
            )
        ]

    lanes = _lane_lookup(profile)
    seen_class_names = {d.class_name for d in detections}
    outputs: List[EventResult] = []
    for idx, cand in enumerate(candidates, start=1):
        lane = lanes[cand.lane_id]
        det_conf = sum(d.confidence for d in cand.detections) / max(len(cand.detections), 1)
        loco_count: Optional[int]
        carriage_count: Optional[int]
        if "locomotive" in seen_class_names:
            loco_count = _count_crossing_ids(detections, lane, cand.start_frame, cand.end_frame, "locomotive")
        else:
            loco_count = None
        if "carriage" in seen_class_names:
            carriage_count = _count_crossing_ids(detections, lane, cand.start_frame, cand.end_frame, "carriage")
        else:
            carriage_count = None
        quality = "ok"
        if cand.direction == "unknown" or cand.speed_mph is None:
            quality = "needs_review"
        outputs.append(
            EventResult(
                video_id=metadata.video_id,
                train_index=idx,
                camera_id=metadata.camera_id,
                track_id=cand.lane_id,
                train_detected=True,
                no_train_flag=False,
                start_time_s=round(cand.start_frame / metadata.fps, 3),
                end_time_s=round(cand.end_frame / metadata.fps, 3),
                duration_s=round((cand.end_frame - cand.start_frame) / metadata.fps, 3),
                direction=cand.direction,
                speed_mph=cand.speed_mph,
                speed_confidence=cand.speed_confidence,
                locomotive_count=loco_count,
                carriage_count=carriage_count,
                company_name="Unknown",
                company_confidence=None,
                detection_confidence=round(det_conf, 3),
                quality_flag=quality,
                processed_at_utc=_now_utc_iso(),
                source_video=str(metadata.source_path),
            )
        )
    return outputs
