# ML Essay Analysis

A cleaned Google Colab research script for analyzing essays linked in Google Sheets.

## What it does

- Loads essay links and labels (`Essay_Link` and `Type`) from Google Sheets.
- Downloads the linked Google Docs text with authorized Google Drive access.
- Computes a GPT-2-large / GPT-2 loss ratio (zero-shot score) and sentence-length burstiness.
- Saves results into a new Google Sheets tab.
- Prints group statistics, AUC, false-positive and false-negative rates when both classes are available, and saves `Academic_Metrics_Dashboard.png`.

## Run in Google Colab

Upload `essay_metrics.py` and `requirements.txt` to your Colab session, then:

```python
!pip install -r requirements.txt
```

Score just the first 10 rows while checking access:

```python
%run essay_metrics.py --sheet-id YOUR_SHEET_ID --tab Sheet1 --pilot
```

Process all rows:

```python
%run essay_metrics.py --sheet-id YOUR_SHEET_ID --tab Sheet1
```

Make sure the signed-in Google account can access the Sheet and all linked documents. The script requests authorization in Colab. The Google Sheet ID is supplied at runtime rather than embedded in this public source file.

## Notes

The `ZeroShot_Score` is a language-model loss ratio, **not** a verified probability of AI authorship. The script assumes the column `Type` contains `Raw` or `Humanized` and uses the negative score for ROC analysis; validate the direction with your data. The in-sample ROC threshold maximizes Youden's J and will be optimistic if used as a reported test result. For research claims, select thresholds on separate validation data and evaluate on held-out data.

This script is intended to run in Colab with a GPU for better performance. The repository contains no essay data or model weights. It has not been run end-to-end against your private Google Sheet.
