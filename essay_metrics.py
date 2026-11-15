"""Evaluate essay links from Google Sheets with zero-shot and burstiness metrics.

Designed for Google Colab with Google Drive and Sheets permissions.
"""

import argparse
import os
import re
import time
from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from google.auth import default
from googleapiclient.discovery import build
from nltk.tokenize import sent_tokenize
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from transformers import AutoModelForCausalLM, AutoTokenizer


CLASS_COLUMN = "Type"
LINK_COLUMN = "Essay_Link"


def extract_doc_id(link):
    if not isinstance(link, str):
        return None
    match = re.search(r"/d/([A-Za-z0-9_-]+)", link)
    return match.group(1) if match else None


def download_doc_text(drive_service, doc_id):
    try:
        request = drive_service.files().export_media(
            fileId=doc_id, mimeType="text/plain"
        )
        return request.execute().decode("utf-8")
    except Exception as exc:
        print(f"Unable to download document {doc_id}: {exc}")
        return None


def calculate_metrics(text, tokenizer, observer, performer, device):
    if not isinstance(text, str) or len(text.strip()) < 10:
        return np.nan, np.nan

    sentences = sent_tokenize(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    burstiness = (
        float(np.std(lengths) / np.mean(lengths))
        if len(lengths) > 1 and np.mean(lengths) > 0
        else 0.0
    )

    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tokens = tokens.to(device)
    if tokens.input_ids.size(1) < 10:
        return np.nan, burstiness

    with torch.inference_mode():
        loss_observer = observer(tokens.input_ids, labels=tokens.input_ids).loss.item()
        loss_performer = performer(tokens.input_ids, labels=tokens.input_ids).loss.item()

    score = loss_observer / loss_performer if loss_performer > 0 else np.nan
    return float(score), burstiness


def evaluate_rows(df, drive_service, tokenizer, observer, performer, device,
                  *, pilot=False, delay=1.0):
    if LINK_COLUMN not in df.columns:
        raise ValueError(f"Google Sheet is missing required column: {LINK_COLUMN}")

    scores = np.full(len(df), np.nan)
    burstiness = np.full(len(df), np.nan)
    limit = min(10, len(df)) if pilot else len(df)

    for i in range(limit):
        doc_id = extract_doc_id(df.iloc[i][LINK_COLUMN])
        if not doc_id:
            print(f"Row {i + 1}/{limit}: missing Google Doc link")
            continue

        text = download_doc_text(drive_service, doc_id)
        if text is not None:
            scores[i], burstiness[i] = calculate_metrics(
                text, tokenizer, observer, performer, device
            )
        print(f"Row {i + 1}/{limit}: score={scores[i]:.4f}, "
              f"burstiness={burstiness[i]:.4f}")
        if delay > 0 and i + 1 < limit:
            time.sleep(delay)

    result = df.copy()
    result["ZeroShot_Score"] = scores
    result["Burstiness_CV"] = burstiness
    return result


def research_metrics(df):
    if CLASS_COLUMN not in df.columns:
        raise ValueError(f"Google Sheet is missing required column: {CLASS_COLUMN}")

    clean = df.dropna(subset=["ZeroShot_Score", CLASS_COLUMN]).copy()
    clean[CLASS_COLUMN] = clean[CLASS_COLUMN].astype(str).str.strip().str.lower()
    clean = clean[clean[CLASS_COLUMN].isin(("raw", "humanized"))].copy()
    clean["Is_Raw"] = (clean[CLASS_COLUMN] == "raw").astype(int)

    raw = clean[clean["Is_Raw"] == 1]
    humanized = clean[clean["Is_Raw"] == 0]
    metrics = {
        "total": len(clean),
        "raw": len(raw),
        "humanized": len(humanized),
        "mean_separation": abs(
            raw["ZeroShot_Score"].mean() - humanized["ZeroShot_Score"].mean()
        ),
        "mean_burstiness_raw": raw["Burstiness_CV"].mean(),
        "mean_burstiness_humanized": humanized["Burstiness_CV"].mean(),
    }

    if not raw.empty and not humanized.empty:
        y_true = clean["Is_Raw"].to_numpy()
        y_scores = -clean["ZeroShot_Score"].to_numpy()
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        threshold = thresholds[int(np.argmax(tpr - fpr))]
        prediction = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        metrics.update({
            "auc": float(roc_auc_score(y_true, y_scores)),
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "fnr": fn / (fn + tp) if fn + tp else np.nan,
            "roc": (fpr, tpr),
        })
    return clean, metrics


def save_dashboard(clean, metrics, output):
    if "auc" not in metrics:
        print("Skipping plots: both Raw and Humanized examples are required.")
        return

    sns.set_context("paper", font_scale=1.3)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("Essay Transformation Analysis", fontsize=20)

    try:
        sns.kdeplot(data=clean, x="ZeroShot_Score", hue=CLASS_COLUMN,
                    fill=True, common_norm=False, ax=axes[0])
    except (ValueError, TypeError):
        sns.histplot(data=clean, x="ZeroShot_Score", hue=CLASS_COLUMN,
                     element="step", ax=axes[0])
    axes[0].set_title("Cross-Perplexity Distribution")

    fpr, tpr = metrics["roc"]
    axes[1].plot(fpr, tpr, label=f"ROC (AUC = {metrics['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], linestyle="--", label="Chance")
    axes[1].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curve")
    axes[1].legend()

    sns.boxplot(data=clean, x=CLASS_COLUMN, y="Burstiness_CV", ax=axes[2])
    axes[2].set_title("Sentence-Length Burstiness")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output}")


def show_summary(metrics):
    print("\nESSAY RESEARCH METRICS")
    print(f"Essays evaluated: {metrics['total']}")
    print(f"Raw: {metrics['raw']} | Humanized: {metrics['humanized']}")
    print(f"Mean score separation: {metrics['mean_separation']:.4f}")
    print(f"Mean burstiness (Raw): {metrics['mean_burstiness_raw']:.4f}")
    print(f"Mean burstiness (Humanized): {metrics['mean_burstiness_humanized']:.4f}")
    if "auc" in metrics:
        print(f"ROC AUC: {metrics['auc']:.4f}")
        print(f"FPR: {metrics['fpr']:.1%} | FNR: {metrics['fnr']:.1%}")
    else:
        print("ROC metrics unavailable without both classes.")


def authenticate_google():
    try:
        from google.colab import auth
    except ImportError as exc:
        raise RuntimeError("Run this script in Google Colab to authenticate.") from exc
    auth.authenticate_user()
    credentials, _ = default()
    return gspread.authorize(credentials), build("drive", "v3", credentials=credentials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet-id", default=os.getenv("ESSAY_SHEET_ID"),
                        help="Google Sheets document ID; or set ESSAY_SHEET_ID")
    parser.add_argument("--tab", default="Sheet1")
    parser.add_argument("--pilot", action="store_true", help="Score only the first 10 rows")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--output", default="Academic_Metrics_Dashboard.png")
    args = parser.parse_args()
    if not args.sheet_id:
        parser.error("Set --sheet-id or ESSAY_SHEET_ID")

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("Tip: enable a GPU in Colab for faster model inference.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    observer = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device).eval()
    performer = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

    sheets, drive_service = authenticate_google()
    worksheet = sheets.open_by_key(args.sheet_id).worksheet(args.tab)
    df = pd.DataFrame(worksheet.get_all_records())
    print(f"Loaded {len(df)} rows from '{args.tab}'")
    result = evaluate_rows(df, drive_service, tokenizer, observer, performer,
                           device, pilot=args.pilot, delay=args.delay)

    new_tab = f"Final_Metrics_{time.strftime('%Y%m%d_%H%M%S')}"
    output_data = result.fillna("")
    out_sheet = sheets.open_by_key(args.sheet_id).add_worksheet(
        title=new_tab, rows=len(result) + 5, cols=len(result.columns) + 2
    )
    out_sheet.update([output_data.columns.tolist()] + output_data.values.tolist(),
                     range_name="A1")
    print(f"Saved score columns to Google Sheets tab '{new_tab}'")

    clean, metrics = research_metrics(result)
    show_summary(metrics)
    save_dashboard(clean, metrics, Path(args.output))


if __name__ == "__main__":
    main()
