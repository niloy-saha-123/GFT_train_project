from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

try:
    import gspread  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    gspread = None


@dataclass(slots=True)
class SheetsConfig:
    enabled: bool
    spreadsheet_id: Optional[str]
    worksheet_name: str
    credentials_path: Optional[Path]
    retries: int = 4
    backoff_seconds: float = 1.5


class GoogleSheetsWriter:
    def __init__(self, config: SheetsConfig):
        self.config = config

    def _get_sheet(self):
        if not self.config.enabled:
            return None
        if gspread is None:
            raise RuntimeError("gspread not installed. Install gspread + google-auth.")
        if not self.config.spreadsheet_id or not self.config.credentials_path:
            raise ValueError("Sheets enabled but spreadsheet_id/credentials_path missing.")
        client = gspread.service_account(filename=str(self.config.credentials_path))
        spreadsheet = client.open_by_key(self.config.spreadsheet_id)
        try:
            return spreadsheet.worksheet(self.config.worksheet_name)
        except gspread.WorksheetNotFound:
            return spreadsheet.add_worksheet(title=self.config.worksheet_name, rows=2000, cols=40)

    @staticmethod
    def _make_key(row: Dict[str, object], key_fields: Sequence[str]) -> str:
        return "::".join(str(row.get(name, "")) for name in key_fields)

    def upsert_rows(self, rows: List[Dict[str, object]], key_fields: Sequence[str]) -> bool:
        if not rows:
            return True
        if not self.config.enabled:
            LOGGER.info("Sheets disabled. Skipping remote sync.")
            return True

        attempt = 0
        while True:
            try:
                sheet = self._get_sheet()
                assert sheet is not None
                existing = sheet.get_all_records()
                existing_keys = {
                    self._make_key(row, key_fields): idx + 2
                    for idx, row in enumerate(existing)
                }
                columns = list(rows[0].keys())
                if not existing:
                    sheet.append_row(columns)
                for row in rows:
                    key = self._make_key(row, key_fields)
                    values = [row.get(col, "") for col in columns]
                    if key in existing_keys:
                        sheet.update(f"A{existing_keys[key]}", [values])
                    else:
                        sheet.append_row(values)
                return True
            except Exception as exc:  # pragma: no cover - network side-effect
                attempt += 1
                if attempt > self.config.retries:
                    LOGGER.error("Google Sheets upsert failed after retries: %s", exc)
                    return False
                sleep_s = self.config.backoff_seconds * (2 ** (attempt - 1))
                LOGGER.warning("Sheets upsert failed (attempt %s). Retrying in %.2fs: %s", attempt, sleep_s, exc)
                time.sleep(sleep_s)


def write_rows_to_excel(rows: Iterable[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    try:
        frame.to_excel(output_path, index=False)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency at runtime
        LOGGER.warning("Excel export skipped, dependency missing: %s", exc)
    return output_path


def write_rows_to_csv(rows: Iterable[Dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    frame.to_csv(output_path, index=False)
    return output_path
