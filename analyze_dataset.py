"""Score available essays from the repository JSONL dataset.

Example:
  python analyze_dataset.py --pilot --output-dir outputs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset_audit import read_dataset
from essay_metrics import calculate_metrics, research_metrics, save_dashboard, show_summary
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(args):
    rows = [r for r in read_dataset(args.dataset_dir)
            if r.get("text_status") == "available" and isinstance(r.get("text"), str)]
    if args.pilot:
        rows = rows[:10]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No available essay text. Run recover_missing_text.py.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Using CPU; processing many essays may be slow.")

    import nltk
    for name in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{name}")
        except LookupError:
            nltk.download(name, quiet=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    observer = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device).eval()
    performer = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    results = []
    for i, row in enumerate(rows, 1):
        score, burstiness = calculate_metrics(
            row["text"], tokenizer, observer, performer, device
        )
        results.append({"essay_id": row["essay_id"], "Type": row["label"],
                        "ZeroShot_Score": score, "Burstiness_CV": burstiness})
        if i % 10 == 0:
            print(f"Scored {i}/{len(rows)}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "essay_scores.csv", index=False)
    clean, metrics = research_metrics(frame)
    show_summary(metrics)
    save_dashboard(clean, metrics, output / "Academic_Metrics_Dashboard.png")
    (output / "metrics_summary.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "roc"},
                   default=lambda x: float(x) if np.isscalar(x) else str(x), indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
