from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List

import cv2

from .types import VideoMetadata


CAMERA_ID_PATTERN = re.compile(r"(camera_\d+)", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def infer_camera_id(filename: str) -> str:
    match = CAMERA_ID_PATTERN.search(filename)
    if match:
        return match.group(1).lower()
    return "camera_unknown"


def extract_video_metadata(video_path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0:
        fps = 30.0

    return VideoMetadata(
        video_id=video_path.stem,
        source_path=video_path,
        file_hash=sha256_file(video_path),
        camera_id=infer_camera_id(video_path.name),
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_s=frame_count / fps if frame_count and fps else 0.0,
    )


def sample_frames(video_path: Path, frame_indices: Iterable[int]) -> List[tuple[int, "cv2.typing.MatLike"]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    samples = []
    for frame_index in sorted(set(int(i) for i in frame_indices if i >= 0)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if ok:
            samples.append((frame_index, frame))
    capture.release()
    return samples
