from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from rail_video_intelligence.pipeline import (  # noqa: E402
    RailVideoPipeline,
    SheetsConfig,
    load_camera_profile,
    load_pipeline_settings,
    parse_pipeline_settings,
)


INBOX_DIR = Path("data/inbox")
INBOX_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(config_path: Path, profile_path: Path) -> RailVideoPipeline:
    raw = load_pipeline_settings(config_path)
    settings = parse_pipeline_settings(raw.get("pipeline", raw))
    sheets_raw = raw.get("sheets", {})
    sheets = SheetsConfig(
        enabled=bool(sheets_raw.get("enabled", False)),
        spreadsheet_id=sheets_raw.get("spreadsheet_id"),
        worksheet_name=sheets_raw.get("worksheet_name", "events"),
        credentials_path=Path(sheets_raw["credentials_path"]) if sheets_raw.get("credentials_path") else None,
        retries=int(sheets_raw.get("retries", 4)),
        backoff_seconds=float(sheets_raw.get("backoff_seconds", 1.5)),
    )
    profile = load_camera_profile(profile_path)
    return RailVideoPipeline(
        settings=settings,
        camera_profile=profile,
        output_root=Path(raw.get("output_root", "outputs/v1")),
        sheets_config=sheets,
    )


def save_uploads(uploaded_files) -> list[Path]:
    saved = []
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = INBOX_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    for item in uploaded_files:
        target = run_dir / item.name
        target.write_bytes(item.getbuffer())
        saved.append(target)
    return saved


def main() -> None:
    st.set_page_config(page_title="Rail Video Ops v1", layout="wide")
    st.title("Rail Video Ops v1 Dashboard")
    st.caption("Upload videos, process detections, update Google Sheets, cleanup source videos on success.")

    with st.sidebar:
        st.header("Config")
        config_path = Path(st.text_input("Pipeline config", value="configs/pipeline.yaml"))
        profile_path = Path(st.text_input("Camera profile", value="configs/camera_profiles/camera_01.yaml"))
        run_prefix = st.text_input("Run prefix", value="dashboard")
        delete_on_success = st.checkbox("Delete source video on successful sheet sync", value=True)
        export_excel = st.checkbox("Export Excel per run", value=True)

    uploaded = st.file_uploader(
        "Upload one or more videos",
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Upload video files to start.")
        return

    st.write(f"Queued videos: {len(uploaded)}")
    queue_df = pd.DataFrame({"filename": [file.name for file in uploaded], "size_bytes": [file.size for file in uploaded]})
    st.dataframe(queue_df, use_container_width=True)

    if st.button("Process Queue", type="primary"):
        try:
            pipeline = build_pipeline(config_path=config_path, profile_path=profile_path)
        except Exception as exc:
            st.error(f"Config load failed: {exc}")
            return

        saved_paths = save_uploads(uploaded)
        progress = st.progress(0)
        rows = []
        previews = []
        for idx, video_path in enumerate(saved_paths, start=1):
            run_name = f"{run_prefix}_{idx:03d}_{video_path.stem}"
            try:
                result = pipeline.process_video(
                    video_path=video_path,
                    run_name=run_name,
                    delete_on_success=delete_on_success,
                    export_excel=export_excel,
                )
                rows.append(
                    {
                        "video": video_path.name,
                        "events": len(result.events),
                        "sheet_sync_success": result.sheet_sync_success,
                        "deleted_source": result.deleted_source,
                        "summary_json": str(result.output_json_path),
                    }
                )
                if result.preview_video_path:
                    previews.append((video_path.name, result.preview_video_path))
            except Exception as exc:
                rows.append(
                    {
                        "video": video_path.name,
                        "events": 0,
                        "sheet_sync_success": False,
                        "deleted_source": False,
                        "summary_json": f"error: {exc}",
                    }
                )
            progress.progress(idx / len(saved_paths))

        result_df = pd.DataFrame(rows)
        st.subheader("Processing Results")
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download run summary JSON",
            data=json.dumps(rows, indent=2),
            file_name=f"{run_prefix}_summary.json",
            mime="application/json",
        )

        if previews:
            st.subheader("Preview Videos")
            for name, preview_path in previews:
                st.markdown(f"**{name}**")
                st.video(str(preview_path))


if __name__ == "__main__":
    main()
