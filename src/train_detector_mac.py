#!/usr/bin/env python3
"""
train_detector_mac.py
─────────────────────
Local Mac M4 train detection pipeline.
Uses Apple MPS (Metal Performance Shaders) for GPU acceleration.

Usage (run from repo root: GFT_train_project/):
    python src/train_detector_mac.py
    python src/train_detector_mac.py --video data/raw/video_1.MOV
    python src/train_detector_mac.py --video-dir data/raw
    python src/train_detector_mac.py --no-display
    python src/train_detector_mac.py --video /path/to/video.MOV --no-save

Install dependencies once:
    pip install ultralytics easyocr opencv-python Pillow numpy tqdm
"""

import argparse
import collections
import pathlib
import time
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FINETUNED_MODEL_PATH = pathlib.Path(__file__).parent / 'train_detector_v1.pt'

CARRIAGE_LENGTH_M  = 16.0
KMH_TO_MPH         = 0.621371
FISHEYE_K1         = -0.38
YOLO_CONF          = 0.25
YOLO_IOU           = 0.45
MOTION_THRESHOLD   = 0.015

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = _PROJECT_ROOT / 'data' / 'raw'
OUTPUT_DIR    = _PROJECT_ROOT / 'outputs'
SAVE_VIDEO    = True
FRAME_SKIP    = 1

KNOWN_COMPANIES = {
    'acela':  'Amtrak Acela',
    'amtrak': 'Amtrak',
    'marc':   'MARC',
}

# ── OCR — Colab v3 style: center band + frame gap only; EasyOCR every eligible read
# (no MSER gate, no locomotive skip — names often on lead power)
OCR_CENTER_ZONE        = 0.20   # middle 60% horizontally, same as v3 (0.20–0.80)
OCR_FRAME_GAP          = 6      # min frames between OCR passes (v3 used 6)
OCR_STOP_CONFIDENCE    = 0.95   # rarely stop trying to read more text
OCR_MAX_CALLS          = 60     # safety cap per train session

# How long no detections = train has left
TRAIN_GONE_THRESH_S = 1.5
TRAIN_GONE_EXTRA_S  = 2.0

# Module-level ocr_reader — assigned in main() before any processing
ocr_reader = None


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


# ══════════════════════════════════════════════════════════════════════════════
# FISHEYE
# ══════════════════════════════════════════════════════════════════════════════

def build_fisheye_maps(h: int, w: int, k1: float = FISHEYE_K1):
    cx, cy = w / 2.0, h / 2.0
    f      = max(w, h) * 0.72
    K      = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    D      = np.array([[k1], [0.0], [0.0], [0.0]], dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    return map1, map2


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Motion gate
# ══════════════════════════════════════════════════════════════════════════════

class TrainPresenceDetector:
    def __init__(self, threshold: float = MOTION_THRESHOLD):
        self.bg_sub   = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=40, detectShadows=False)
        self.threshold = threshold
        self.history   = collections.deque(maxlen=8)

    def update(self, frame) -> bool:
        fg    = self.bg_sub.apply(frame)
        ratio = np.count_nonzero(fg) / (frame.shape[0] * frame.shape[1])
        self.history.append(ratio)
        return float(np.mean(self.history)) > self.threshold


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Direction detector
# ══════════════════════════════════════════════════════════════════════════════

class DirectionDetector:
    def __init__(self, frame_w: int):
        self.frame_w   = frame_w
        self.direction = None
        self.cx_buf    = collections.deque(maxlen=10)

    def update(self, cx: float):
        self.cx_buf.append(cx)
        if self.direction is None and len(self.cx_buf) >= 5:
            delta = self.cx_buf[-1] - self.cx_buf[0]
            if abs(delta) > 15:
                self.direction = 'L→R' if delta > 0 else 'R→L'
        return self.direction

    def reset(self):
        self.direction = None
        self.cx_buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Counter with numbered units
# ══════════════════════════════════════════════════════════════════════════════

class CarriageCounter:
    """
    Each unique ByteTrack ID crossing the line gets a sequential label:
        Consist-1, Consist-2, Loco-1, etc.
    Speed uses first/last crossing frame timestamps.
    """
    def __init__(self, frame_w: int, frame_h: int, dir_det: DirectionDetector):
        self.frame_w       = frame_w
        self.frame_h       = frame_h
        self.dir_det       = dir_det
        self.line_x        = frame_w // 2
        self.crossed_ids   = set()
        self.unit_labels   = {}   # track_id → "Consist-1", "Loco-2" etc.
        self.count         = 0
        self.consist_count = 0
        self.loco_count    = 0
        self.t_first       = None
        self.t_last        = None

    def update(self, detections: list, frame_num: int):
        for d in detections:
            tid = d.get('track_id')
            if tid is None:
                continue
            cx        = (d['x1'] + d['x2']) / 2.0
            direction = self.dir_det.direction or 'L→R'
            if tid not in self.crossed_ids:
                crossed = (
                    (direction == 'L→R' and cx > self.line_x) or
                    (direction == 'R→L' and cx < self.line_x)
                )
                if crossed:
                    self.crossed_ids.add(tid)
                    self.count += 1
                    cls_name = d.get('cls_name', 'consist')
                    if cls_name == 'locomotive':
                        self.loco_count += 1
                        label = f"Loco-{self.loco_count}"
                    else:
                        self.consist_count += 1
                        label = f"Consist-{self.consist_count}"
                    self.unit_labels[tid] = label
                    if self.t_first is None:
                        self.t_first = frame_num
                    self.t_last = frame_num

    def get_label(self, tid) -> str:
        return self.unit_labels.get(tid, f"ID:{tid}")

    def compute_speed(self, fps: float):
        if self.t_first is None or self.t_last is None or self.count < 2:
            return None, None
        transit_frames = self.t_last - self.t_first
        if transit_frames < fps * 0.3:
            return None, None
        total_m   = self.count * CARRIAGE_LENGTH_M
        transit_s = transit_frames / fps
        kmh       = (total_m / transit_s) * 3.6
        if not (5 < kmh < 400):
            return None, None
        return round(kmh * KMH_TO_MPH, 1), round(kmh, 1)

    def reset(self):
        self.crossed_ids.clear()
        self.unit_labels.clear()
        self.count = self.consist_count = self.loco_count = 0
        self.t_first = self.t_last = None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Smart OCR — fixed, less restrictive
# ══════════════════════════════════════════════════════════════════════════════

class TrainOCR:
    """Match Colab v3: crop → resize → sharpen → EasyOCR (no MSER); all classes."""

    def __init__(self, frame_w: int, frame_h: int):
        self.frame_w         = frame_w
        self.frame_h         = frame_h
        self.ocr_x1          = int(frame_w * OCR_CENTER_ZONE)
        self.ocr_x2          = int(frame_w * (1.0 - OCR_CENTER_ZONE))
        self.votes           = collections.Counter()
        self.company         = None
        self.conf            = 0.0
        self._last_ocr_frame = -20
        self._total_calls    = 0

    def should_run(self, cx: float, frame_num: int) -> bool:
        if self._total_calls >= OCR_MAX_CALLS:
            return False
        if self.conf >= OCR_STOP_CONFIDENCE:
            return False
        if not (self.ocr_x1 < cx < self.ocr_x2):
            return False
        if frame_num - self._last_ocr_frame < OCR_FRAME_GAP:
            return False
        return True

    def run(
        self, frame, x1: int, y1: int, x2: int, y2: int, frame_num: int,
    ) -> list:
        self._last_ocr_frame = frame_num
        crop_h = max(1, int((y2 - y1) * 0.65))
        crop   = frame[y1: y1 + crop_h, x1: x2]
        if crop.size == 0 or crop.shape[0] < 12:
            return []

        if crop.shape[0] < 80:
            s    = 80 / crop.shape[0]
            crop = cv2.resize(crop, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)

        k    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        crop = cv2.filter2D(crop, -1, k)

        self._total_calls += 1
        try:
            results = ocr_reader.readtext(crop, detail=1, paragraph=False)
            return [txt.strip().lower()
                    for (_, txt, conf) in results
                    if conf > 0.40 and len(txt.strip()) >= 3]
        except Exception:
            return []

    def add(self, texts: list):
        for t in texts:
            self.votes[t] += 1

    def identify(self):
        if not self.votes:
            return None, 0.0

        best_company, best_score = None, 0
        for text, cnt in self.votes.most_common(15):
            for key, company in KNOWN_COMPANIES.items():
                if key in text.lower() or text.lower() in key:
                    if cnt > best_score:
                        best_score   = cnt
                        best_company = company
                        self.conf    = min(1.0, cnt / 5.0)
        if best_company:
            self.company = best_company
            return best_company, self.conf

        best_ratio, best_co = 0.0, None
        for text, cnt in self.votes.most_common(8):
            for key, company in KNOWN_COMPANIES.items():
                score = SequenceMatcher(None, text.lower(), key).ratio()
                if score > best_ratio and score > 0.55:
                    best_ratio = score
                    best_co    = company
                    self.conf  = score
        if best_co:
            self.company = best_co
            return best_co, self.conf

        top, _ = self.votes.most_common(1)[0]
        return top.upper(), 0.25

    def reset(self):
        self.votes.clear()
        self.company         = None
        self.conf            = 0.0
        self._last_ocr_frame = -20
        self._total_calls    = 0


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Renderer
# ══════════════════════════════════════════════════════════════════════════════

class Renderer:
    F      = cv2.FONT_HERSHEY_SIMPLEX
    GREEN  = (0,   200,  80)
    WHITE  = (255, 255, 255)
    GRAY   = (20,   20,  20)
    YELLOW = (0,   210, 220)
    CLASS_COLORS = {
        'consist':    (255, 140,   0),
        'locomotive': (0,    60, 220),
    }

    def draw(self, frame, dets, counter, ocr, speed_mph, speed_kmh,
             direction, frame_num, fps, proc_fps, train_index):
        out  = frame.copy()
        h, w = out.shape[:2]

        cv2.line(out, (counter.line_x, 0), (counter.line_x, h), self.GREEN, 2)
        cv2.putText(out, 'COUNT LINE', (counter.line_x + 5, 22),
                    self.F, 0.45, self.GREEN, 1)

        for d in dets:
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            tid      = d.get('track_id', '?')
            cls_name = d.get('cls_name', 'consist')
            color    = self.CLASS_COLORS.get(cls_name, (255, 140, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = counter.get_label(tid)
            cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                        self.F, 0.40, color, 1)

        spd_str = (f"{speed_mph:.1f} mph  ({speed_kmh:.1f} km/h)"
                   if speed_mph else "calculating...")
        lines = [
            f"Train #{train_index}  |  Frame {frame_num}  [{proc_fps:.0f} fps]",
            f"Direction : {direction or 'detecting...'}",
            f"Consists  : {counter.consist_count}",
            f"Locos     : {counter.loco_count}",
            f"Speed     : {spd_str}",
            f"Name      : {ocr.company or 'detecting...'}",
            f"OCR calls : {ocr._total_calls}/{OCR_MAX_CALLS}",
        ]
        for i, line in enumerate(lines):
            y = 20 + i * 22
            cv2.rectangle(out, (4, y - 14), (360, y + 6), self.GRAY, -1)
            cv2.putText(out, line, (7, y), self.F, 0.46, self.WHITE, 1)

        cv2.putText(out, f"t={frame_num / max(fps, 1):.1f}s",
                    (w - 120, h - 10), self.F, 0.45, self.YELLOW, 1)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN SESSION — one per train
# ══════════════════════════════════════════════════════════════════════════════

class TrainSession:
    """
    All state for one train passing through the frame.
    Completely isolated from other trains — own counter, OCR, direction.
    """
    def __init__(self, frame_w: int, frame_h: int, train_index: int):
        self.train_index = train_index
        self.dir_det     = DirectionDetector(frame_w)
        self.counter     = CarriageCounter(frame_w, frame_h, self.dir_det)
        self.ocr         = TrainOCR(frame_w, frame_h)
        self.result      = None

    def finalise(self, fps: float) -> dict:
        company, conf = self.ocr.identify()
        if not company:
            company = 'Unknown'
            conf    = 0.0
        speed_mph, speed_kmh = self.counter.compute_speed(fps)
        self.result = {
            'train_index':        self.train_index,
            'company_name':       company,
            'company_confidence': round(conf, 2),
            'consist_count':      self.counter.consist_count,
            'locomotive_count':   self.counter.loco_count,
            'total_units':        self.counter.count,
            'unit_labels':        dict(self.counter.unit_labels),
            'direction':          self.dir_det.direction or 'unknown',
            'speed_mph':          speed_mph,
            'speed_kmh':          speed_kmh,
            'ocr_votes':          dict(self.ocr.votes.most_common(10)),
            'ocr_calls':          self.ocr._total_calls,
        }
        return self.result

    def print_result(self):
        r = self.result
        print(f"\n{'═'*55}")
        print(f"  TRAIN #{r['train_index']} RESULTS")
        print(f"{'═'*55}")
        print(f"  Name        : {r['company_name']}  (conf: {r['company_confidence']:.0%})")
        print(f"  Consists    : {r['consist_count']}")
        print(f"  Locomotives : {r['locomotive_count']}")
        print(f"  Total units : {r['total_units']}")
        if r['unit_labels']:
            print(f"  Units       : {', '.join(r['unit_labels'].values())}")
        print(f"  Direction   : {r['direction']}")
        if r['speed_mph']:
            print(f"  Speed       : {r['speed_mph']} mph  ({r['speed_kmh']} km/h)")
        else:
            print(f"  Speed       : not enough data")
        print(f"  OCR raw     : {list(r['ocr_votes'].items())[:5]}")
        print(f"{'═'*55}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_video(
    video_path: pathlib.Path,
    yolo_model,
    device: str,
    show_live: bool          = True,
    save_output: bool        = SAVE_VIDEO,
    output_dir: pathlib.Path = OUTPUT_DIR,
    frame_skip: int          = FRAME_SKIP,
) -> list:
    """Returns list of result dicts — one per train detected."""

    video_path = pathlib.Path(video_path)
    assert video_path.exists(), f"Video not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open: {video_path}"

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    GONE_THRESH = int(fps * TRAIN_GONE_THRESH_S)
    EXTRA_WAIT  = int(fps * TRAIN_GONE_EXTRA_S)

    print(f"\n{'='*60}")
    print(f"  Video  : {video_path.name}")
    print(f"  Size   : {w}×{h}  |  {fps:.1f} FPS  |  {total_frm} frames ({total_frm/fps:.1f}s)")
    print(f"  Device : {device.upper()}")
    print(f"{'='*60}\n")

    map1, map2 = build_fisheye_maps(h, w)

    writer   = None
    out_path = None
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (video_path.stem + '_annotated.mp4')
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    cls_names   = yolo_model.names
    presence    = TrainPresenceDetector()
    renderer    = Renderer()

    # ── Multi-train state machine ─────────────────────────────────────────────
    state       = 'WAITING'
    gone_frames = 0
    train_index = 0
    session     = None       # current TrainSession — None when no train present
    all_results = []

    speed_mph = None
    speed_kmh = None
    frame_num = 0
    t_start   = time.perf_counter()
    proc_fps  = 0.0

    # Blank session just for rendering when no train present
    blank_session = TrainSession(w, h, 0)

    if show_live:
        cv2.namedWindow('Train Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Train Detector', min(w, 1280), min(h, 720))

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % frame_skip != 0:
            if writer:
                writer.write(raw)
            if show_live:
                cv2.imshow('Train Detector', cv2.remap(raw, map1, map2, cv2.INTER_LINEAR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        frame  = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR)
        active = presence.update(frame)
        dets   = []

        if active or state in ('TRAIN_IN_FRAME', 'TRAIN_GONE'):
            results = yolo_model.track(
                frame,
                persist = True,
                classes = None,
                conf    = YOLO_CONF,
                iou     = YOLO_IOU,
                tracker = 'bytetrack.yaml',
                verbose = False,
            )

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    tid      = int(box.id[0]) if box.id is not None else None
                    cls_idx  = int(box.cls[0])
                    cls_name = cls_names.get(cls_idx, 'consist')
                    dets.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': float(box.conf[0]),
                        'track_id': tid,
                        'cls_name': cls_name,
                    })

            if dets:
                # ── New train arriving ────────────────────────────────────────
                if state == 'TRAIN_GONE' and session is not None:
                    result = session.finalise(fps)
                    all_results.append(result)
                    session.print_result()
                    train_index += 1
                    session      = TrainSession(w, h, train_index)
                    speed_mph    = None
                    speed_kmh    = None
                    print(
                        f"\nTrain #{train_index} at frame {frame_num} "
                        f"(t={frame_num/fps:.1f}s) — new session after prior train"
                    )
                elif state == 'WAITING':
                    train_index += 1
                    session      = TrainSession(w, h, train_index)
                    speed_mph    = None
                    speed_kmh    = None
                    print(
                        f"\nTrain #{train_index} detected at frame {frame_num} "
                        f"(t={frame_num/fps:.1f}s)"
                    )

                state       = 'TRAIN_IN_FRAME'
                gone_frames = 0

                for d in dets:
                    session.dir_det.update((d['x1'] + d['x2']) / 2.0)
                session.counter.update(dets, frame_num)

                for d in dets:
                    cx = (d['x1'] + d['x2']) / 2.0
                    if session.ocr.should_run(cx, frame_num):
                        texts = session.ocr.run(
                            frame,
                            d['x1'], d['y1'], d['x2'], d['y2'],
                            frame_num,
                        )
                        session.ocr.add(texts)
                if session.ocr.company is None and len(session.ocr.votes) >= 2:
                    session.ocr.identify()

                speed_mph, speed_kmh = session.counter.compute_speed(fps)

            else:
                if state == 'TRAIN_IN_FRAME':
                    gone_frames += 1
                    if gone_frames >= GONE_THRESH:
                        state = 'TRAIN_GONE'

        # ── Train gone — wait extra then finalise ─────────────────────────────
        if state == 'TRAIN_GONE':
            gone_frames += 1
            if gone_frames >= GONE_THRESH + EXTRA_WAIT:
                if session is not None:
                    result = session.finalise(fps)
                    all_results.append(result)
                    session.print_result()
                # Reset — ready for next train
                state       = 'WAITING'
                gone_frames = 0
                session     = None
                speed_mph   = None
                speed_kmh   = None
                print(f"\n⏳ Waiting for next train...")

        # Render
        render_session = session if session is not None else blank_session
        elapsed        = time.perf_counter() - t_start
        proc_fps       = frame_num / elapsed if elapsed > 0 else 0

        annotated = renderer.draw(
            frame, dets,
            render_session.counter,
            render_session.ocr,
            speed_mph, speed_kmh,
            render_session.dir_det.direction,
            frame_num, fps, proc_fps,
            train_index if state != 'WAITING' else '—',
        )

        if show_live:
            cv2.imshow('Train Detector', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user.")
                break
            elif key == ord('s'):
                snap = output_dir / f'snapshot_train{train_index}_f{frame_num}.jpg'
                cv2.imwrite(str(snap), annotated)
                print(f"Snapshot: {snap}")

        if writer:
            writer.write(annotated)

    # End of video — finalise any session still in progress
    if session is not None and state in ('TRAIN_IN_FRAME', 'TRAIN_GONE'):
        result = session.finalise(fps)
        all_results.append(result)
        session.print_result()

    cap.release()
    if writer:
        writer.release()
    if show_live:
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start

    print(f"\n{'═'*55}")
    print(f"  VIDEO SUMMARY — {video_path.name}")
    print(f"{'═'*55}")
    print(f"  Trains detected : {len(all_results)}")
    print(f"  Process time    : {total_time:.1f}s  ({frame_num/total_time:.1f} fps avg)")
    if out_path:
        print(f"  Output video    : {out_path}")
    for r in all_results:
        spd = f"{r['speed_mph']} mph" if r['speed_mph'] else "N/A"
        print(f"  Train #{r['train_index']:>2}  {r['company_name']:15s}  "
              f"{r['consist_count']} consists  {r['locomotive_count']} locos  "
              f"{spd}  {r['direction']}")
    print(f"{'═'*55}\n")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

_VIDEO_EXTS = {'.mov', '.mp4', '.m4v'}


def list_videos_in_dir(video_dir: pathlib.Path) -> list[pathlib.Path]:
    if not video_dir.is_dir():
        return []
    return sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )


def main():
    global ocr_reader, FISHEYE_K1

    parser = argparse.ArgumentParser(description='Train detector — Mac M4')
    parser.add_argument('--video',       type=str,   default=None)
    parser.add_argument('--video-dir',   type=str,   default=None)
    parser.add_argument('--model',       type=str,   default=str(FINETUNED_MODEL_PATH))
    parser.add_argument('--no-display',  action='store_true')
    parser.add_argument('--no-save',     action='store_true')
    parser.add_argument('--output-dir',  type=str,   default=str(OUTPUT_DIR))
    parser.add_argument('--frame-skip',  type=int,   default=1)
    parser.add_argument('--k1',          type=float, default=FISHEYE_K1)
    args = parser.parse_args()

    if args.video and args.video_dir:
        parser.error('Use either --video or --video-dir, not both.')

    if args.video:
        video_paths = [pathlib.Path(args.video)]
    else:
        vdir = pathlib.Path(args.video_dir) if args.video_dir else RAW_VIDEO_DIR
        if not vdir.is_dir():
            print(f"❌ Folder not found: {vdir}"); sys.exit(1)
        video_paths = list_videos_in_dir(vdir)
        if not video_paths:
            print(f"❌ No videos in: {vdir}"); sys.exit(1)

    model_path = pathlib.Path(args.model)
    output_dir = pathlib.Path(args.output_dir)

    for vp in video_paths:
        if not vp.exists():
            print(f"❌ Video not found: {vp}"); sys.exit(1)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}"); sys.exit(1)

    device = get_device()
    print(f"\n{'─'*50}")
    print(f"  Train Detector — Mac Local Pipeline")
    print(f"{'─'*50}")
    print(f"  Device  : {device.upper()}")
    print(f"  Model   : {model_path.name}")
    if len(video_paths) == 1:
        print(f"  Video   : {video_paths[0].name}")
    else:
        print(f"  Batch   : {len(video_paths)} videos")
    print(f"  Display : {'OFF' if args.no_display else 'ON (press q to quit, s to snapshot)'}")
    print(f"  Save    : {'OFF' if args.no_save else 'ON'}")
    print(f"{'─'*50}\n")

    print("Loading YOLO model...")
    t0         = time.perf_counter()
    yolo_model = YOLO(str(model_path))
    yolo_model.to(device)
    print(f"✅ YOLO loaded in {time.perf_counter()-t0:.1f}s  |  Classes: {yolo_model.names}")

    print("Loading EasyOCR...")
    t0              = time.perf_counter()
    ocr_reader_inst = easyocr.Reader(['en'], gpu=False, verbose=False)
    ocr_reader      = ocr_reader_inst
    print(f"✅ EasyOCR loaded in {time.perf_counter()-t0:.1f}s")

    FISHEYE_K1 = args.k1

    for i, vp in enumerate(video_paths, 1):
        if len(video_paths) > 1:
            print(f"\n>>> [{i}/{len(video_paths)}] {vp.name}")
        process_video(
            video_path  = vp,
            yolo_model  = yolo_model,
            device      = device,
            show_live   = not args.no_display,
            save_output = not args.no_save,
            output_dir  = output_dir,
            frame_skip  = args.frame_skip,
        )


if __name__ == '__main__':
    main()