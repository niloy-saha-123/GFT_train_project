from rail_video_intelligence.pipeline.sheets import GoogleSheetsWriter, SheetsConfig


class _FakeSheet:
    def __init__(self):
        self.rows = []
        self.updated = []

    def get_all_records(self):
        if not self.rows:
            return []
        header = self.rows[0]
        data = []
        for row in self.rows[1:]:
            data.append({key: value for key, value in zip(header, row)})
        return data

    def append_row(self, row):
        self.rows.append(row)

    def update(self, _cell, values):
        self.updated.extend(values)


def test_upsert_retries_and_succeeds(monkeypatch):
    config = SheetsConfig(
        enabled=True,
        spreadsheet_id="sheet",
        worksheet_name="events",
        credentials_path=None,
        retries=2,
        backoff_seconds=0.0,
    )
    writer = GoogleSheetsWriter(config)
    fake_sheet = _FakeSheet()
    attempts = {"count": 0}

    def fake_get_sheet():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("temporary error")
        return fake_sheet

    monkeypatch.setattr(writer, "_get_sheet", fake_get_sheet)
    rows = [{"video_id": "v1", "train_index": 1, "direction": "right_to_left"}]
    assert writer.upsert_rows(rows, key_fields=("video_id", "train_index")) is True
    assert attempts["count"] == 2
