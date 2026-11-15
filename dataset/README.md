# Labeled essay dataset

The source spreadsheet contains **2,809 rows**: **1,400 Raw** and **1,409 Humanized**. This repository represents all rows in **57 JSONL partitions**, retaining `essay_id`, `source_row`, `label`, `word_count_reported`, `text`, and `text_status`.

**Current completeness: 300 essays with usable text; 2,509 missing.** The source spreadsheet's `Original_Essay_Text` column uses the literal `INVALID_LINK` for many entries, even though sampled linked Docs are accessible. To avoid passing these placeholders as essay prose, unresolved records now contain `text: null`, `text_status: needs_document_fetch`.

`python dataset_audit.py` summarizes the status. `python dataset_audit.py --strict` fails until all missing texts are recovered. The Colab script `recover_missing_text.py` uses authorized Google Drive export to fetch each missing text by its source row and save repaired JSONL partitions. Example: `%run recover_missing_text.py --sheet-id YOUR_SHEET_ID --limit 10`, then remove `--limit 10` to recover all. It does not alter the source Sheet or push files automatically. Check output and permissions before publishing.

No Google Doc URL, writer name, or document ID appears in the records, but essay text may itself contain identifying details. **Public reuse, redistribution and privacy compliance have not been independently verified.** Raw/Humanized labels are copied from the source, not independently assessed, and the unpaired row ordering should not be assumed to establish matched pairs.
