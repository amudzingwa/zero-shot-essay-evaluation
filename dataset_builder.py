"""Export labeled Google Docs essays from a Google Sheet into a local dataset.

Run in Google Colab. Essay text is private by default; review permissions
and personal information before publishing a dataset.
"""

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

LINK_COLUMN = "Essay_Link"
LABEL_COLUMN = "Type"
FIELDS = ("essay_id", "Type", "text", "word_count", "char_count")


def extract_google_doc_id(link):
    if not isinstance(link, str):
        return None
    match = re.search(
        r"https?://docs\.google\.com/document/d/([A-Za-z0-9_-]+)",
        link.strip(),
    )
    return match.group(1) if match else None


def canonical_label(value):
    label = str(value).strip().lower()
    if label == "raw":
        return "Raw"
    if label == "humanized":
        return "Humanized"
    return None


def build_records(sheet_rows, load_text, limit=None, pause_seconds=0):
    """Create dataset records without copying source URLs or document IDs.

    load_text receives a Google Doc ID and should return a string.
    """
    rows = sheet_rows[:limit] if limit is not None else sheet_rows
    records = []
    errors = Counter()
    seen_ids = set()
    seen_content = set()

    for index, row in enumerate(rows, start=1):
        label = canonical_label(row.get(LABEL_COLUMN))
        doc_id = extract_google_doc_id(row.get(LINK_COLUMN))
        if label is None:
            errors["unsupported_label"] += 1
            continue
        if doc_id is None:
            errors["invalid_doc_link"] += 1
            continue
        if doc_id in seen_ids:
            errors["duplicate_doc_link"] += 1
            continue
        seen_ids.add(doc_id)
        try:
            text = load_text(doc_id)
        except Exception:
            errors["download_failed"] += 1
            continue
        if not isinstance(text, str) or len(text.strip()) < 10:
            errors["empty_or_short_text"] += 1
            continue
        text = text.strip()
        if text in seen_content:
            errors["duplicate_text"] += 1
            continue
        seen_content.add(text)
        records.append({
            "essay_id": f"essay_{len(records) + 1:06d}",
            "Type": label,
            "text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
        })
        if pause_seconds and index < len(rows):
            time.sleep(pause_seconds)

    return records, dict(errors)


def save_records(records, errors, output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "essays.csv"
    jsonl_path = folder / "essays.jsonl"
    manifest_path = folder / "manifest.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "total_essays": len(records),
        "labels": dict(Counter(row["Type"] for row in records)),
        "skipped": errors,
        "fields": list(FIELDS),
        "publication_status": "private_pending_permission_and_privacy_review",
        "note": "No authors, source document IDs, or Google Docs URLs are included.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return csv_path, jsonl_path, manifest_path


def load_google_sheet(sheet_id, tab):
    try:
        from google.colab import auth
        import gspread
        from google.auth import default
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            "Run in Google Colab after installing requirements.txt"
        ) from exc

    auth.authenticate_user()
    credentials, _ = default()
    client = gspread.authorize(credentials)
    worksheet = client.open_by_key(sheet_id).worksheet(tab)
    headers = worksheet.row_values(1)
    for expected in (LINK_COLUMN, LABEL_COLUMN):
        if expected not in headers:
            raise ValueError(f"Sheet is missing the {expected!r} column")
    rows = worksheet.get_all_records()
    drive = build("drive", "v3", credentials=credentials)

    def load_text(doc_id):
        request = drive.files().export_media(
            fileId=doc_id, mimeType="text/plain"
        )
        return request.execute().decode("utf-8")

    return rows, load_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sheet-id", default=os.environ.get("ESSAY_SHEET_ID"),
        help="Google Sheet ID (or set ESSAY_SHEET_ID)",
    )
    parser.add_argument("--tab", default="Sheet1")
    parser.add_argument("--output-dir", default="datasets/private")
    parser.add_argument("--pilot", action="store_true", help="Read first 10 rows only")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    if not args.sheet_id:
        parser.error("Provide --sheet-id or ESSAY_SHEET_ID")

    rows, load_text = load_google_sheet(args.sheet_id, args.tab)
    records, errors = build_records(
        rows, load_text,
        limit=10 if args.pilot else None,
        pause_seconds=max(0, args.delay),
    )
    paths = save_records(records, errors, args.output_dir)
    print(f"Read {len(records)} essays from {min(len(rows), 10) if args.pilot else len(rows)} sheet rows.")
    print(f"Labels: {dict(Counter(row['Type'] for row in records))}")
    print(f"Skipped: {errors}")
    for path in paths:
        print(f"Saved: {path}")
    print("These files contain essay text; review permission, privacy, and licensing before sharing.")


if __name__ == "__main__":
    main()
