# Financial-Sentiment-Classification Model Development

Part of the AI Financial Research Assistant project. This document covers the model development work completed so far: dataset, fine-tuning approach, models compared, and results.

## Objective

Build a sentiment classifier that labels financial text (news, filings) as **positive**, **negative**, or **neutral**, to power the sentiment tracking and dashboard features of a broader system.

## Dataset

**Financial PhraseBank** (Malo et al., 2014) — ~4,846 sentences from financial news and press releases, annotated by 16 finance-background annotators for sentiment.

- Source used: `atrost/financial_phrasebank` on Hugging Face (a pre-converted, split version of the original dataset)
- Subset: 50%+ annotator agreement (`sentences_50agree`), consistent with the standard FinBERT benchmark setup
- Splits: 3,100 train / 776 validation / 970 test — pre-defined, not custom-split
- Labels: `0 = negative`, `1 = neutral`, `2 = positive`

## Models fine-tuned

Two base models were fine-tuned and compared, alongside an off-the-shelf baseline:

| Model | Base | Approach |
|---|---|---|
| DistilBERT | `distilbert-base-uncased` | General-English base, fine-tuned from scratch on Financial PhraseBank |
| FinBERT (fine-tuned) | `ProsusAI/finbert` | Finance-pretrained base; original classification head discarded and retrained on Financial PhraseBank, using the same process as DistilBERT |
| FinBERT (off-the-shelf) | `ProsusAI/finbert` | No additional training — the publisher's original classification head, used as a baseline for comparison |

## Fine-tuning methodology

- **Framework:** PyTorch, via Hugging Face `transformers` (`Trainer` API) and `datasets`
- **Preprocessing:** each model's own tokenizer used to tokenize the `sentence` field, with truncation; dynamic padding per batch via `DataCollatorWithPadding`
- **Training/selection:** validation-set evaluation after every epoch; best checkpoint (by validation loss) retained automatically (`load_best_model_at_end`), to guard against overfitting on later epochs
- **Held-out test set:** never used during training or checkpoint selection — reserved solely for final, unbiased evaluation
- **Hyperparameter search:** Bayesian optimization via **Optuna**, integrated through `Trainer.hyperparameter_search()`. Search space:
  - Learning rate: 5e-6 to 5e-5 (log scale)
  - Epochs: 2–3 (capped based on observed early plateau in manual runs)
  - Batch size: 8 / 16 / 32
  - Weight decay: 0.0–0.1
  - Objective: maximize validation accuracy
  - The winning configuration was then retrained in a final full run and evaluated on the held-out test set.

### Why FinBERT's original head was replaced

`ProsusAI/finbert` ships with its own pretrained classification head, trained by its original authors with a different label ordering than this project's scheme. To keep the comparison fair and the work fully reproducible, that head was discarded (`ignore_mismatched_sizes=True`) and a fresh head — matching this project's exact label order — was trained from scratch on the same data and process used for DistilBERT.

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| DistilBERT (fine-tuned, default hyperparameters) | 82.9% | First baseline; general-English base |
| FinBERT (fine-tuned, default/manual hyperparameters) | 85.4% | Same process as DistilBERT, finance-pretrained base |
| FinBERT (fine-tuned, Optuna-selected hyperparameters) | **86.1%** | Best of 5 search trials, retrained and evaluated on test set |
| FinBERT (off-the-shelf, no additional training) | 86.5% | Publisher's original head, used as upper-reference baseline |

**Optuna search results (5 trials, optimizing validation accuracy):**

| Trial | Learning Rate | Epochs | Batch Size | Weight Decay | Validation Accuracy |
|---|---|---|---|---|---|
| 0 (best) | 7.36e-06 | 2 | 8 | 0.0103 | 87.2% |
| 1 | 6.61e-06 | 2 | 32 | 0.0238 | 80.9% |
| 2 | 1.86e-05 | 3 | 32 | 0.0732 | 87.0% |
| 3 | 1.50e-05 | 2 | 16 | 0.0142 | 86.9% |
| 4 | 2.30e-05 | 2 | 32 | 0.0789 | 85.4% |

**Observations:**

- Domain-pretrained FinBERT outperformed general-purpose DistilBERT by ~2.5 points under identical fine-tuning conditions, confirming the value of finance-specific pretraining for this task.
- Hyperparameter search via Optuna improved FinBERT's test accuracy from 85.4% (manual guess) to 86.1% — closing most, though not all, of the gap to the off-the-shelf baseline.
- The winning learning rate found by the search (7.36e-06) was notably lower than the manually-chosen 1e-5, and well below DistilBERT's 2e-5 — consistent with the expectation that a finance-pretrained base needs gentler fine-tuning to avoid overwriting its existing domain knowledge (catastrophic forgetting).
- The off-the-shelf FinBERT head still slightly outperformed this project's best tuned version (86.5% vs. 86.1%), suggesting the original authors' training setup (likely more data — possibly the full, unfiltered agreement levels — and/or more extensive tuning) generalizes marginally better than what's achievable fine-tuning on the 50%-agreement subset alone. This is reported honestly rather than adjusted in favor of the in-project model.
- All fine-tuning results are in line with published benchmarks for this dataset (DistilBERT ~82%, BERT ~86%, FinBERT ~91%, per external comparative studies).

## Reproducibility notes

- Training was run on Google Colab (free-tier T4 GPU).
- `load_best_model_at_end=True` combined with per-epoch validation evaluation was the primary safeguard against overfitting within a single run — validation loss, not just accuracy, informed checkpoint selection.
- All models were evaluated only once on the test set, after all training and hyperparameter decisions were finalized, to keep the reported numbers unbiased.

## Status / next steps

- [x] DistilBERT baseline fine-tuned and evaluated
- [x] FinBERT fine-tuned (manual hyperparameters) and evaluated
- [x] FinBERT off-the-shelf baseline evaluated for comparison
- [x] Optuna hyperparameter search implemented and run (5 trials)
- [x] Final Optuna-tuned model evaluated on test set (86.1%)
- [ ] Final model selected and moved into `services/sentiment-service/model/`
- [ ] Model wrapped in FastAPI service and containerized

---

*This README documents model development only. Service architecture, API details, and deployment notes will be documented separately once the sentiment service is built.*
