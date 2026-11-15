# ML Essay Analysis

Research code and a labeled essay dataset sourced from **MASTER AI GENERATED ESSAYS**, tab **Sheet1**.

## Dataset status

| Item | Rows |
|---|---:|
| Source essay records | **2,809** |
| Raw labels | **1,400** |
| Humanized labels | **1,409** |
| Usable essay text in repository | **300** |
| Text requiring retrieval from linked Docs | **2,509** |

The `dataset/` folder contains 57 JSONL files with every source record. Each record includes `essay_id`, `source_row`, `label`, `word_count_reported`, `text`, and `text_status`. Records whose source Sheet text was `INVALID_LINK` are explicitly marked `text: null` and `text_status: needs_document_fetch`. **Do not treat these as complete essays.** The Google Docs links were not copied into the dataset.

### Audit and recovery

Run in a Python environment from the repository root:

```bash
python dataset_audit.py
```

For full recovery, open the repository in **Google Colab**, authenticate with an account that can access the linked Docs, and run:

```python
!pip install -r requirements.txt
%run recover_missing_text.py --sheet-id 156HvNm21Z9-ot-VyOIvIcteUMf6II26bKQqPbgvevhQ --limit 10
```

Check the ten repaired records, then rerun without `--limit 10` to attempt all remaining missing texts. The recovery process can take time, may hit Drive quotas, and can be resumed. It edits *local JSONL files only*. You need to review the text, publication rights, and privacy before pushing the recovered dataset.

Check completion:

```bash
python dataset_audit.py --strict
```

### Score available essays locally

Use Google Colab and enable a GPU under Runtime settings:

```python
!pip install -r requirements.txt
%run analyze_dataset.py --pilot
```

For all available texts remove `--pilot`. Scores are saved in `outputs/essay_scores.csv`, along with `metrics_summary.json` and a dashboard image.

### Original Sheets-based workflow

`essay_metrics.py` can still read directly from Google Sheets, export Google Docs text, compute zero-shot loss ratios and sentence-length burstiness, save scores to a new tab, and report AUC, FPR and FNR when both labels are available:

```python
%run essay_metrics.py --sheet-id YOUR_SHEET_ID --tab Sheet1 --pilot
```

## Research and publication caveats

These are source-provided `Raw`/`Humanized` labels, not independently verified AI-authorship judgments. The GPT-2-large/GPT-2 loss ratio is *not* a probability of AI authorship. Validation and test should be separated by originating prompt or essay family to avoid leakage. Selecting an ROC threshold on evaluation data can overstate reported performance.

The dataset is present in a **public GitHub repository**. Essay publication permissions and potential identifying details have **not** been individually audited; verify rights and privacy before using or further distributing the text. Model weights and original source Google Doc links are not included.

See [dataset/README.md](dataset/README.md) and [dataset/manifest.json](dataset/manifest.json).
