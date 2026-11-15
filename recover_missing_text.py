"""Recover missing essay text from linked Google Docs using Google Colab.

Run in Colab with the same Google account that can open the spreadsheet.
This script writes recovered dataset files locally; review before git push.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from dataset_audit import read_dataset, summarize


def doc_id_from_url(link):
    m = re.search(r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)", str(link))
    return m.group(1) if m else None


def get_clients():
    from google.colab import auth
    from google.auth import default
    from googleapiclient.discovery import build
    import gspread
    auth.authenticate_user()
    credentials, _ = default()
    return gspread.authorize(credentials), build("drive", "v3", credentials=credentials)


def recover(dataset_dir, sheet_id, tab="Sheet1", delay=1, limit=None):
    sheets, drive = get_clients()
    ws = sheets.open_by_key(sheet_id).worksheet(tab)
    headers = ws.row_values(1)
    if "Essay_Link" not in headers:
        raise ValueError("Source sheet needs an Essay_Link column")
    col = headers.index("Essay_Link") + 1
    links = ws.col_values(col)
    folder = Path(dataset_dir)
    files = sorted(folder.glob("essays_*.jsonl"))
    recovered, failures, processed = 0, 0, 0
    for file in files:
        records = [json.loads(s) for s in file.read_text(encoding="utf-8").splitlines() if s.strip()]
        changed = False
        for record in records:
            if record.get("text_status") == "available" and isinstance(record.get("text"), str):
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1
            source_row = int(record["source_row"])
            link = links[source_row - 1] if source_row - 1 < len(links) else ""
            doc_id = doc_id_from_url(link)
            if not doc_id:
                failures += 1
                continue
            try:
                response = drive.files().export_media(fileId=doc_id, mimeType="text/plain").execute()
                text = response.decode("utf-8").strip()
                if len(text) < 10:
                    raise ValueError("Document text is empty")
                record["text"] = text
                record["text_status"] = "available"
                recovered += 1
                changed = True
            except Exception as exc:
                failures += 1
                print(f"Unable to retrieve essay {record['essay_id']}: {exc}")
            if delay > 0:
                time.sleep(delay)
        if changed:
            file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                            encoding="utf-8")
            print(f"Updated {file}")
        if limit is not None and processed >= limit:
            break
    print(f"Recovered: {recovered}; Failed: {failures}; Remaining: "
          f"{summarize(read_dataset(folder))['counts'].get('missing', 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-id", default=os.environ.get("ESSAY_SHEET_ID"))
    parser.add_argument("--tab", default="Sheet1")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--delay", type=float, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not args.sheet_id:
        parser.error("--sheet-id or ESSAY_SHEET_ID is required")
    recover(args.dataset_dir, args.sheet_id, args.tab, args.delay, args.limit)
