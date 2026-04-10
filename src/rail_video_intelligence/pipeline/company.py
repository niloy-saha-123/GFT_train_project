from __future__ import annotations

from typing import Dict, List, Optional

import cv2

from .types import DetectionRecord, EventResult, VideoMetadata

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


COMPANY_TOKEN_MAP: Dict[str, str] = {
    "acela": "Amtrak Acela",
    "amtrak": "Amtrak",
    "csx": "CSX",
    "norfolk": "Norfolk Southern",
    "ns": "Norfolk Southern",
    "bnsf": "BNSF",
    "union pacific": "Union Pacific",
}


def _extract_text(frame, bbox) -> str:
    if pytesseract is None:
        return ""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    crop = frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.lower().strip()


def _classify_company(text: str) -> tuple[str, Optional[float]]:
    if not text:
        return "Unknown", None
    for token, label in COMPANY_TOKEN_MAP.items():
        if token in text:
            conf = 0.75 if len(token) >= 5 else 0.6
            return label, conf
    return "Unknown", 0.35


def enrich_company_labels(
    metadata: VideoMetadata,
    events: List[EventResult],
    detections: List[DetectionRecord],
) -> List[EventResult]:
    if not events:
        return events
    capture = cv2.VideoCapture(str(metadata.source_path))
    if not capture.isOpened():
        return events

    try:
        train_detections = [d for d in detections if d.class_name == "train"]
        for event in events:
            if event.no_train_flag:
                continue
            mid_frame = int((((event.start_time_s or 0.0) + (event.end_time_s or 0.0)) / 2.0) * metadata.fps)
            event_dets = [
                d
                for d in train_detections
                if d.lane_id == event.track_id
                and event.start_time_s is not None
                and event.end_time_s is not None
                and event.start_time_s <= d.time_s <= event.end_time_s
            ]
            if not event_dets:
                continue
            chosen = min(event_dets, key=lambda d: abs(d.frame_index - mid_frame))
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(chosen.frame_index, 0))
            ok, frame = capture.read()
            if not ok:
                continue
            text = _extract_text(frame, chosen.bbox_xyxy)
            name, conf = _classify_company(text)
            event.company_name = name
            event.company_confidence = conf
    finally:
        capture.release()
    return events
