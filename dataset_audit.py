"""Audit labeled essay JSONL files and report usable text coverage."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_dataset(folder):
    for path in sorted(Path(folder).glob("essays_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize(records):
    counters = Counter()
    labels = Counter()
    available_labels = Counter()
    seen = set()
    for row in records:
        essay_id = row.get("essay_id")
        if essay_id in seen:
            counters["duplicate_ids"] += 1
        seen.add(essay_id)
        label = row.get("label")
        labels[label] += 1
        counters["total"] += 1
        text = row.get("text")
        if isinstance(text, str) and len(text.strip()) >= 10 and text.strip() not in {"INVALID_LINK", "ERROR"}:
            counters["available"] += 1
            available_labels[label] += 1
        else:
            counters["missing"] += 1
    return {"counts": dict(counters), "labels": dict(labels),
            "available_labels": dict(available_labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero while any essay text remains missing")
    args = parser.parse_args()
    report = summarize(read_dataset(args.dataset_dir))
    print(json.dumps(report, indent=2))
    if args.strict and (report["counts"].get("missing", 0) or report["counts"].get("duplicate_ids", 0)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
