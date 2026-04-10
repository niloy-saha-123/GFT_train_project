# Rail Video Intelligence

Rail Video Intelligence is a computer vision and machine learning project for extracting operational train data from trackside video. The system is intended to process uploaded rail videos and return structured observations such as train speed, pass time, travel direction, consist count, locomotive count, and train type. The long-term goal is to turn raw video into a reliable event stream that can support reporting, analytics, alerting, and future downstream modeling.

## Project Goals

This project is designed to answer a practical set of questions from rail videos:

- How fast is the train moving?
- When did the train enter and leave the scene?
- Which direction is it moving?
- How many total consists/cars are present?
- How many locomotives are present?
- What type of train is passing?
- What additional metadata can be captured later as the system matures?

## Target Outputs

For each processed video, the system should eventually produce a structured record similar to the following:

```json
{
  "video_id": "camera_01_2026_04_09_173000",
  "train_detected": true,
  "start_time_s": 12.4,
  "end_time_s": 74.8,
  "duration_s": 62.4,
  "direction": "northbound",
  "direction_cardinal": "north",
  "speed_mph": 47.2,
  "speed_confidence": 0.86,
  "locomotive_count": 2,
  "consist_count": 84,
  "train_type": "intermodal",
  "train_type_confidence": 0.78,
  "camera_id": "camera_01",
  "track_id": "mainline_east",
  "processing_version": "v0.1.0"
}
```

## Problem Definition

Trackside rail video is difficult to interpret automatically because conditions vary by camera angle, lighting, weather, motion blur, occlusion, train length, and train composition. A robust solution needs more than one model. In practice, this is best handled as a pipeline with multiple stages rather than one monolithic model.

### Core tasks

1. Train presence detection
2. Train direction estimation
3. Train speed estimation
4. Car/consist counting
5. Locomotive counting
6. Train type classification
7. Event timestamp extraction

### Future tasks

- Railcar subtype classification
- Logo/operator identification
- Hazard placard detection
- Track occupancy duration
- Multi-track disambiguation
- Audio-assisted inference
- Exception detection for stopped or broken-up trains

## Recommended System Design

The most practical first version is a multi-stage computer vision pipeline:

1. Video ingestion
2. Frame extraction or stream decoding
3. Object detection on train segments, locomotives, and cars
4. Multi-object tracking across frames
5. Trackline calibration and motion estimation
6. Event aggregation into one train pass
7. Structured result generation

### Why this approach

Different outputs depend on different signals:

- Speed depends on distance calibration and time.
- Direction depends on motion trajectory relative to camera geometry.
- Consist and locomotive count depend on detection and tracking quality.
- Train type depends on composition patterns and visual appearance.

Because of this, a hybrid design is usually more accurate and easier to debug than a single end-to-end model.

## Proposed Technical Approach

### 1. Detection

Use an object detector to identify:

- Locomotives
- Freight cars
- Passenger cars
- Caboose/end-of-train marker when visible
- Whole-train regions when useful

Strong baseline options:

- YOLOv8 / YOLO11 for fast training and inference
- RT-DETR for stronger transformer-based detection if latency allows
- Grounding DINO or similar only for bootstrapping labels, not as primary production inference

### 2. Tracking

Use object tracking to maintain identity across frames:

- ByteTrack
- BoT-SORT
- OC-SORT

Tracking is critical for:

- Avoiding duplicate counts
- Estimating motion over time
- Building train-level events from frame-level detections

### 3. Speed Estimation

Speed should not be predicted as a pure black-box label if real-world distance can be calibrated. A better solution is:

- Define reference points in camera view
- Estimate scale in feet/meters per pixel at track plane
- Track locomotive or car position through time
- Convert displacement per frame into real speed

Possible methods:

- Manual camera calibration with known trackside landmarks
- Homography / perspective correction
- Virtual tripwire method using two known points
- Optical flow as secondary signal for robustness checks

### 4. Direction Estimation

Direction can be derived from track geometry and object motion:

- Determine dominant motion vector
- Map movement to camera-specific direction metadata
- Convert to operational labels like northbound/southbound or eastbound/westbound

Important note:
cardinal direction cannot be inferred universally from pixels alone. Each camera needs metadata defining how image motion maps to real-world direction.

### 5. Consist and Locomotive Counting

Two complementary strategies should be supported:

- Detection + tracking count
- Event-line crossing count at virtual gate

Gate-crossing often gives more stable counts than raw per-frame counting, especially for long trains.

### 6. Train Type Classification

Train type can be inferred from:

- Lead equipment appearance
- Car mix distribution
- Average car shapes
- Train length
- Presence of containers, tank cars, hoppers, autoracks, passenger coaches, etc.

Suggested labels for initial phase:

- Intermodal
- Coal
- Mixed freight
- Manifest
- Passenger
- Tank train
- Grain
- Autorack
- Maintenance / work train
- Unknown

### 7. Event Aggregation

Frame-level outputs must be merged into train-level events:

- Detect pass start
- Detect pass end
- Merge temporary missed detections
- Count unique units
- Produce final confidence values

## Data Strategy

This project will depend heavily on data quality. Before model training, build data workflows deliberately.

### Required data sources

- Raw rail videos from fixed cameras
- Camera metadata
- Track metadata
- Ground-truth observations when available
- Optional manual logs from operators or dispatch systems

### Important metadata per camera

- Camera ID
- GPS location
- Mount height
- Camera angle
- Lens/FOV
- Frame rate
- Resolution
- Track orientation
- Mapping from image motion to cardinal direction
- Known physical reference distances in scene

### Labeling requirements

Recommended annotation types:

- Bounding boxes for locomotives
- Bounding boxes for railcars
- Train-level class labels
- Entry/exit timestamps
- Direction labels
- True speed labels when available

Possible tooling:

- CVAT
- Label Studio
- Roboflow
- Supervisely

## Suggested Project Phases

### Phase 1: Feasibility and data audit

- Collect representative sample videos
- Group by camera, weather, day/night, train type
- Validate frame rates and video quality
- Define annotation schema
- Create baseline README, environment, and repo structure

### Phase 2: Baseline train event detector

- Detect whether a train is present
- Detect train pass start/end
- Estimate direction
- Produce simple pass summaries

### Phase 3: Counting models

- Train locomotive detector
- Train railcar detector
- Add tracking and stable counting logic

### Phase 4: Speed estimation

- Add camera calibration
- Build tripwire or trajectory-based speed estimator
- Benchmark error against ground truth

### Phase 5: Train type classifier

- Add train-level classifier
- Aggregate per-frame visual features into one event-level prediction

### Phase 6: Production hardening

- Batch inference pipeline
- Confidence thresholds
- Quality checks
- Failure handling
- Monitoring and data review loops

## Evaluation Plan

Each task needs separate evaluation metrics.

### Detection metrics

- mAP@50
- mAP@50:95
- Precision
- Recall

### Tracking metrics

- MOTA
- IDF1
- Track fragmentation rate

### Counting metrics

- Absolute count error
- Mean absolute percentage error
- Exact match rate

### Speed metrics

- MAE in mph or km/h
- RMSE
- Percentage within ±5 mph
- Percentage within ±10% relative error

### Classification metrics

- Accuracy
- Macro F1
- Confusion matrix

### Event metrics

- Start time error
- End time error
- Direction accuracy

## Risks and Constraints

This problem has several real-world failure modes:

- Night video and glare
- Rain, fog, snow, and shadows
- Camera shake
- Occlusion from poles, vegetation, or crossings
- Multiple tracks in frame
- Partial train visibility
- Variable train lengths
- Similar-looking car types
- Missing calibration data for speed

These risks should be handled explicitly in dataset design and evaluation.

## Repository Plan

Recommended project layout:

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── annotations/
├── notebooks/
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── tracking/
│   ├── inference/
│   └── utils/
├── artifacts/
├── outputs/
└── tests/
```

## Development Setup

### 1. Create and activate virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Initial Dependencies

This repository starts with packages for:

- computer vision
- model training
- experiment notebooks
- data manipulation
- visualization
- testing
- formatting/linting

See `requirements.txt` for the exact package list.

## Quickstart: v0.1 Baseline Pipeline

Use this baseline to get first-pass outputs for:

- train presence
- pass start/end time
- left/right direction estimate
- optional speed estimate from virtual tripwires

### 1. Put a video in `data/raw/`

Example:

```bash
cp /path/to/video.mp4 data/raw/v0_1.mp4
```

### 2. Update baseline config

Edit `configs/v0_1_baseline.yaml`:

- `source`: video path
- `mode`: `track` or `detect`
- `tripwire_a` and `tripwire_b`: normalized x positions (0 to 1)
- `distance_m`: real-world distance between tripwires in meters
- `camera_id` and `track_id`

### 3. Run pipeline

```bash
python src/rail_video_intelligence/run_v0_1.py --config configs/v0_1_baseline.yaml
```

Optional overrides:

```bash
python src/rail_video_intelligence/run_v0_1.py \
  --config configs/v0_1_baseline.yaml \
  --source data/raw/your_video.mp4 \
  --name first_pass \
  --show

# detection-only mode (no tracker/lap dependency)
python src/rail_video_intelligence/run_v0_1.py \
  --config configs/v0_1_baseline.yaml \
  --mode detect \
  --name detect_only
```

### 4. Check outputs

- Annotated track video:
  `outputs/v0_1/<run_name>/<video_name>.mp4`
- Event summary JSON:
  `outputs/v0_1/<run_name>/event_summary.json`

## Rail Video Ops v1 (Notebook + Dashboard + Google Sheets)

This repository now includes a modular v1 pipeline under `src/rail_video_intelligence/pipeline/`.

### v1 capabilities

- single-video and batch processing
- per-lane event split (supports overlapping opposite-direction trains)
- direction from lane axis projection
- speed from per-lane marker crossing + known real distance
- no-train audit row generation
- Google Sheets upsert by `video_id + train_index`
- Excel/CSV export per run
- source video deletion only after successful Sheets sync

### v1 folder highlights

- `configs/pipeline.yaml`: processing + Sheets settings
- `configs/camera_profiles/camera_01.yaml`: ROI, ignore mask, lanes, speed markers, direction map
- `src/rail_video_intelligence/run_pipeline_v1.py`: CLI pipeline runner
- `streamlit_app.py`: upload dashboard
- `notebooks/01..05_*.ipynb`: notebook workflow

### Run v1 via CLI

```bash
python src/rail_video_intelligence/run_pipeline_v1.py \
  --pipeline-config configs/pipeline.yaml \
  --camera-profile configs/camera_profiles/camera_01.yaml \
  --source data/raw/camera_01_2026_04_09_233235.MOV \
  --run-name pilot_single

# local popup window while processing (desktop only)
python src/rail_video_intelligence/run_pipeline_v1.py \
  --pipeline-config configs/pipeline.yaml \
  --camera-profile configs/camera_profiles/camera_01.yaml \
  --source data/raw/camera_01_2026_04_09_233235.MOV \
  --run-name pilot_show \
  --show-live
```

Batch mode:

```bash
python src/rail_video_intelligence/run_pipeline_v1.py \
  --pipeline-config configs/pipeline.yaml \
  --camera-profile configs/camera_profiles/camera_01.yaml \
  --source-dir data/raw \
  --run-name pilot_batch
```

Delete source file only after successful Sheets sync:

```bash
python src/rail_video_intelligence/run_pipeline_v1.py \
  --pipeline-config configs/pipeline.yaml \
  --camera-profile configs/camera_profiles/camera_01.yaml \
  --source data/raw/camera_01_2026_04_09_233235.MOV \
  --run-name pilot_delete_on_success \
  --delete-on-success
```

### Run dashboard

```bash
streamlit run streamlit_app.py
```

Dashboard features:

- upload one or many videos
- process queue
- show per-video status
- preview annotated output videos
- write Sheets rows
- delete source videos only when sync succeeds
- preview uses demo overlay video (lanes, marker lines, boxes, IDs, event labels)

### Google Sheets setup

Edit `configs/pipeline.yaml`:

- `sheets.enabled: true`
- `sheets.spreadsheet_id: <google_sheet_id>`
- `sheets.worksheet_name: train_events`
- `sheets.credentials_path: /absolute/path/to/service_account.json`

Sheet row schema written by pipeline:

`video_id, train_index, camera_id, track_id, train_detected, no_train_flag, start_time_s, end_time_s, duration_s, direction, speed_mph, speed_confidence, locomotive_count, carriage_count, company_name, company_confidence, detection_confidence, quality_flag, processed_at_utc, source_video, notes`

### GPU selection

Set in `configs/pipeline.yaml`:

- `pipeline.device: auto` (default)
- `pipeline.device: cuda:0` (force first NVIDIA GPU)
- `pipeline.device: cpu` (force CPU)
- `pipeline.show_live: true` for popup windows on local desktop runs
- `pipeline.save_demo_overlay: true` to save `demo_overlay.mp4` with lane/marker visuals

### Notebook sequence

0. `notebooks/00_master_end_to_end.ipynb` (single notebook full flow)
1. `notebooks/01_ingest_qc.ipynb`
2. `notebooks/02_calibration_markers.ipynb`
3. `notebooks/03_detection_tracking.ipynb`
4. `notebooks/04_event_aggregation_speed_count_company.ipynb`
5. `notebooks/05_google_sheets_sync_ops.ipynb`

## First Deliverables

Suggested first engineering tasks:

1. Define video naming convention and metadata schema.
2. Create sample dataset manifest.
3. Build video loader and frame extraction utility.
4. Implement baseline train presence detector.
5. Add camera calibration configuration format.
6. Add inference output schema and JSON exporter.

## Recommended Architecture Decision

For this use case, start with a classical CV + deep learning pipeline, not a large end-to-end vision-language model.

Recommended baseline:

- YOLO detector for locomotives/cars
- ByteTrack for tracking
- Camera-specific calibration for speed
- Rule-based event aggregation
- Lightweight train-type classifier using aggregated visual evidence

Vision-language models can still be useful later for:

- manual review assistance
- zero-shot dataset exploration
- failure case triage
- generating labeling suggestions

They should not be first choice for core production counting and speed estimation.

## Definition of Success

An initial usable version of this system should be able to:

- process uploaded videos automatically
- detect train pass windows
- classify direction reliably per camera
- count locomotives with high precision
- estimate total consist count with acceptable error
- estimate speed within an agreed tolerance band
- export results in machine-readable format

## License

This repository currently includes the `LICENSE` file already present in the project root. Update project licensing terms if commercial deployment requirements change.
