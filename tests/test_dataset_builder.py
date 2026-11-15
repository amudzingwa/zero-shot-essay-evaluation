"""Tests for dataset construction; run python -m unittest discover -s tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset_builder import (
    build_records, canonical_label, extract_google_doc_id, save_records
)


class DatasetBuilderTests(unittest.TestCase):
    def test_link_and_label_validation(self):
        self.assertEqual(
            extract_google_doc_id("https://docs.google.com/document/d/DOC_123/edit"),
            "DOC_123",
        )
        self.assertIsNone(extract_google_doc_id("https://example.com/d/DOC_123"))
        self.assertEqual(canonical_label(" humanized "), "Humanized")
        self.assertIsNone(canonical_label("unknown"))

    def test_records_deduplicate_and_omit_source_links(self):
        sheets = [
            {"Essay_Link": "https://docs.google.com/document/d/abc/edit", "Type": "Raw"},
            {"Essay_Link": "https://docs.google.com/document/d/abc/edit", "Type": "Raw"},
            {"Essay_Link": "https://docs.google.com/document/d/def/edit", "Type": "Humanized"},
            {"Essay_Link": "invalid", "Type": "Raw"},
        ]
        essays = {"abc": "A meaningful essay of sufficient length.",
                  "def": "Another unique essay of sufficient length."}
        rows, skipped = build_records(sheets, essays.__getitem__)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["Type"] for r in rows], ["Raw", "Humanized"])
        self.assertEqual(skipped["duplicate_doc_link"], 1)
        self.assertEqual(skipped["invalid_doc_link"], 1)
        self.assertEqual(set(rows[0]), {"essay_id", "Type", "text", "word_count", "char_count"})
        self.assertNotIn("abc", json.dumps(rows))

    def test_serialized_dataset(self):
        records = [{
            "essay_id": "essay_000001", "Type": "Raw",
            "text": "First line\nSecond line, with comma",
            "word_count": 6, "char_count": 33,
        }]
        with tempfile.TemporaryDirectory() as directory:
            csv_path, jsonl_path, manifest_path = save_records(records, {}, directory)
            with csv_path.open(encoding="utf-8", newline="") as stream:
                parsed = list(csv.DictReader(stream))
            self.assertEqual(parsed[0]["text"], records[0]["text"])
            self.assertEqual(json.loads(jsonl_path.read_text().splitlines()[0])["Type"], "Raw")
            self.assertEqual(json.loads(manifest_path.read_text())["total_essays"], 1)


if __name__ == "__main__":
    unittest.main()
