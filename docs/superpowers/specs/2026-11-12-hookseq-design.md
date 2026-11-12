# HookSeq — Design Spec

**Date:** 2026-11-12
**Owner:** Arush Badhe
**Purpose:** Portfolio project for the Hook Music *Machine Learning Engineering Intern, Recommendation Systems* application.
**Status:** Approved design → ready for implementation plan

---

## 1. Shared understanding

**Goal.** Build one strong, end-to-end music recommender that shows Hook's hiring team the exact skills the posting asks for: data pipelines and feature engineering, implementing and benchmarking recommendation models in PyTorch, ranking, and rigorous evaluation. Wherever it can, it should speak directly to Hook's product.

**What the user said**
- Aim for the best version (no hard time cap).
- Approach B: a sequence-based "what plays / gets liked next" transformer.
- Training runs on **Kaggle notebooks (GPU)**. Code is written locally on a MacBook Air.
- Put real weight on **EDA and preprocessing** and use sound ML concepts throughout.

**Assumptions (correct if wrong)**
- The repo lives in this folder and gets pushed to the user's GitHub.
- Python 3.10+, PyTorch 2.x. Kaggle T4/P100 GPU (16 GB), ~29 GB RAM, 12 h sessions, ~30 GPU-h/week.
- The README and results are the main deliverable a recruiter will see.

**Success criteria**
1. A reproduced SASRec baseline on Yambda-50M lands **within ~15% of the paper's NDCG@10**, which validates the pipeline and evaluation.
2. **HookSeq beats the SASRec baseline on NDCG@10 and Recall@100** under the official protocol on at least one Yambda size, ideally 500M.
3. An **ablation table** shows the effect of each addition (audio, event-type, organic, logQ). Results are reported honestly, including any additions that don't help.
4. An **EDA notebook/report** whose findings visibly drive the preprocessing choices (thresholds, filters, sequence length).
5. The tests pass, the README has a results table plus a "Why this matters for Hook" section, and the Gradio demo runs.

---

## 2. Why this matters for Hook (framing for the README)

| Hook reality | What HookSeq demonstrates |
|---|---|
| Remixing and Explore are session-driven ("what next?") | Next-item sequence modelling on real listening sessions |
| Hook's own feed shapes what users see (feedback loops) | Models and evaluation that use `is_organic`: users choosing vs. being served |
| Catalog keeps growing (new label deals, newly cleared artists) | Audio embeddings in the item representation, so new tracks can be scored from audio alone |
| Artist-first positioning | Artist exposure / long-tail metrics alongside accuracy |
| Likes, skips, dislikes are richer than plays | Event-type tokens; dislikes as signal, not noise |

---

## 3. Data

**Source:** `yandex/yambda` on Hugging Face (Apache-2.0, research use).

| Use | Subset | Size |
|---|---|---|
| Local development, tests, EDA | `flat/50m/multi_event.parquet` | ~47.8M events, 10K users |
| Headline results (Kaggle) | `flat/500m/multi_event.parquet` (or `sequential/500m`) | ~480M events, 100K users |
| Item side info | `embeddings.parquet`, `artist_item_mapping`, `album_item_mapping` | root of repo |

**Schema (multi_event):** `uid`, `item_id`, `timestamp` (uint32, 5-second bins), `is_organic` (uint8), `event_type` (listen/like/dislike/unlike/undislike), `played_ratio_pct`, `track_length_seconds` (null for non-listens).
**Embeddings:** `item_id`, `embed`, `normalized_embed` (CNN audio embeddings; not every track has one).

**Download** goes through `huggingface_hub` / `datasets` into a configurable `DATA_DIR` (Kaggle: `/kaggle/working/data` or an attached Kaggle Dataset cache).

---

## 4. EDA (notebook `notebooks/01_eda.ipynb`, backed by `hookrec.eda`)

Each analysis exists to inform a decision. Every finding is written up together with the choice it led to.

| Analysis | Concept | Decision it informs |
|---|---|---|
| Event-type mix; events per user / per item (log-log plots, power-law fit, Lorenz curve + Gini) | Long-tail / popularity concentration | Min-frequency vocab threshold; need for logQ correction; fairness baseline |
| `played_ratio_pct` distribution, split by organic vs. served | Implicit-feedback noise; implicit vs. explicit feedback | Threshold that defines a "positive listen" (expect ~50%, the paper's Listen+); skip definition |
| Inter-event time gaps per user (histogram of log gaps) | Sessionization | Session gap threshold (e.g. 30 min) used for features and the demo |
| Repeat consumption: share of re-listens, consecutive duplicates | Repeat vs. exploration behaviour | Whether to collapse consecutive duplicates; whether to filter seen items at eval |
| `is_organic` share over time, by user activity and by item popularity | Exposure bias / feedback loops | Organic token; organic-only evaluation slice |
| Temporal drift: daily volume, new items appearing per day, popularity churn | Non-stationarity | Supports temporal split; DecayPop half-life |
| Sequence-length distribution | Truncation trade-off | `max_len` (e.g. 200), sampling of long histories |
| Audio-embedding coverage by popularity; 2-D UMAP/PCA coloured by artist | Content-feature quality | Audio fusion design; fallback for missing embeddings |
| Like/dislike rates vs. played ratio | Label consistency | Multi-task "like" head target; dislike as hard negative |
| Train/test overlap: test-day items unseen in train | Cold-start magnitude | Size of the cold-start evaluation slice |

**Output:** the notebook plus `reports/eda_summary.md` (key plots saved to `reports/figures/`).

---

## 5. Preprocessing pipeline (`hookrec.data`)

These are deterministic, config-driven steps. Each one is unit-tested on synthetic data.

1. **Load** with Polars (lazy scan, column projection) to keep memory low. Cast to compact dtypes.
2. **Clean:** drop events with invalid durations, and de-duplicate identical `(uid,item,timestamp,event_type)` rows. Reconcile like→unlike and dislike→undislike into a final state where needed.
3. **Label events into tokens:** `event_type ∈ {listen_full, listen_skip, like, dislike}` using the EDA-chosen played-ratio threshold. `played_ratio` is also bucketed (e.g. 5 bins) as a feature.
4. **Temporal split (Yambda Global Temporal Split):** train = first 300 days, 30-minute gap, test = the next 1 day. **Validation** = last day of train (with the same gap) for early stopping and tuning. No random splits. A leakage test asserts `max(train.ts) < min(test.ts) - gap`.
5. **Vocabulary / ID remapping:** items with ≥ `min_count` positive interactions **in train only** get an ID embedding; the rest are "ID-less" and represented by audio + artist only. Index 0 = padding. The mapping is saved to disk.
6. **Item side features:** audio embedding (normalized, optionally PCA-reduced to 128-d, fit on train items only), artist index (multi-artist: first/primary). There is a missing-audio indicator plus a learned fallback vector.
7. **Sequence building:** per-user chronological token sequences. Consecutive duplicates are collapsed if the EDA says so. Truncate to the last `max_len`, left-pad. Stored as packed NumPy arrays (`offsets` + `values`) for fast, low-RAM loading.
8. **Targets:** next-item prediction on **positive** events (listen_full or like). Non-positive events stay in the input sequence as context but are not predicted.
9. **Popularity statistics** (from train) for logQ correction, DecayPop and fairness metrics.
10. **Caching:** each stage writes Parquet/NPZ artifacts keyed by a config hash, so Kaggle sessions can resume.

---

## 6. Models (`hookrec.models`)

All models implement a common interface: `fit(train_data)` and `score(user_batch) -> (batch × n_items)` or `topk(user_batch, k)`.

1. **MostPop** and **DecayPop** (exponential time decay; half-life tuned on validation).
2. **iALS** (via the `implicit` library, GPU when available). Confidence weighting `1 + α·count`; factors, regularization and α tuned on validation.
3. **SASRec (baseline):** causal transformer, 2 blocks, 2 heads, d=64–128, learned positional embeddings, dropout 0.2. Item-ID input/output embeddings are shared.
4. **HookSeq** = SASRec plus these additions, each switchable in config for ablations:
   - **Audio fusion:** `item_rep = ID_emb (if in vocab) + MLP(audio) + artist_emb`, with **ID-dropout** (p≈0.2) during training so the audio and artist path learns to stand alone and can serve new and ID-less items.
   - **Event-type and organic embeddings:** added to each input token, along with a played-ratio bucket embedding.
   - **Sampled softmax with logQ correction:** in-batch plus uniformly sampled negatives. Logits are corrected by the log of each item's sampling probability (removes popularity bias in the loss). Dislikes from the user's own history act as extra hard negatives.
   - *(Stretch)* **Multi-task like head:** an auxiliary BCE that predicts whether the next positive event is a like, weighted λ.
   - *(Stretch)* **Time-gap embedding:** a bucketed time-since-previous-event embedding (session awareness).

**Training:** AdamW, cosine schedule with warmup, AMP (fp16) on GPU, gradient clipping, early stopping on validation NDCG@10, fixed seeds. Checkpoints go to `/kaggle/working/checkpoints`.

**Inference:** the user state is the transformer output at the last position, using history up to the test start (state is frozen for the test day, as in Yambda). Scoring is a chunked matmul against all item representations on the GPU, keeping the top-k.

---

## 7. Evaluation (`hookrec.eval`)

- **Protocol:** Yambda's. For each test user, the ground truth is the set of positive items in the test day. Ranking is over the full catalog.
- **Metrics:** NDCG@10/100, Recall@10/100, Coverage@10/100. The implementation is verified against hand-computed toy cases in tests.
- **Slices:**
  - organic-only vs. recommended-only ground truth
  - cold items (unseen in train or ID-less) vs. warm
  - user activity terciles
- **Beyond accuracy:** artist-exposure Gini, artist coverage, share of recommendations in the long tail (bottom 80% of items by popularity).
- **Ablation table:** SASRec → +audio → +event/organic → +logQ → full HookSeq (→ +multi-task).
- **Reporting:** every run writes `results/<run_id>.json` (config + metrics). `hookrec.eval.report` builds the README tables. Paper numbers are included as a reference row.
- **Statistical care:** bootstrap 95% CIs over users for the key metrics in the headline comparison.

---

## 8. Demo (`hookrec.demo`, Gradio)

- Pick a sample user, **or** build a session by searching tracks (by item ID / artist, since Yambda has no titles).
- Show HookSeq's top-10 next tracks next to SASRec's, with each item's popularity and artist, plus an organic/served toggle on the session tokens to show how the signal shifts the predictions.
- "New-track" tab: given an ID-less track (audio only), show the users and sessions it would rank highest for.
- Runs locally from a small exported checkpoint plus a subsample of item representations (fits on CPU).

---

## 9. Repository layout

```
Music reccomendation_system/
├── README.md                 # results table, Hook framing, how to run
├── pyproject.toml            # package + deps (torch, polars, numpy, pyarrow, implicit, gradio, matplotlib, pyyaml, pytest)
├── configs/                  # data.yaml, sasrec.yaml, hookseq.yaml, ials.yaml, ablations/*.yaml
├── src/hookrec/
│   ├── data/                 # download.py, load.py, clean.py, split.py, vocab.py, sequences.py, features.py
│   ├── eda/                  # stats.py, plots.py
│   ├── models/               # popularity.py, ials.py, sasrec.py, hookseq.py, losses.py
│   ├── train/                # trainer.py, datasets.py (torch Dataset/DataLoader)
│   ├── eval/                 # metrics.py, protocol.py, slices.py, fairness.py, report.py
│   ├── demo/                 # app.py
│   └── utils/                # config.py, seed.py, logging.py
├── notebooks/                # 01_eda, 02_baselines, 03_train_hookseq_kaggle, 04_results
├── scripts/                  # run_preprocess.py, run_train.py, run_eval.py
├── tests/                    # synthetic-data unit tests per module
├── reports/                  # eda_summary.md, figures/
├── results/                  # run JSONs (small, committed)
└── docs/superpowers/{specs,plans}/
```

Large data and checkpoints are git-ignored. Kaggle notebooks run `git clone <repo> && pip install -e .` and then call the scripts.

---

## 10. Testing strategy

- `tests/fixtures.py` generates a tiny synthetic Yambda-shaped dataset (≈50 users, 200 items, all event types, organic flags, a few items without audio).
- Unit tests:
  - cleaning and dedupe
  - labeling thresholds
  - temporal split, including no leakage
  - vocab and ID-less handling
  - sequence packing / truncation / padding
  - metric correctness on hand-computed examples
  - logQ loss sanity (equals plain softmax when sampling probabilities are uniform)
  - model forward shapes and causal masking (no attention to the future)
  - an overfit-one-batch smoke test
- CI-free: `pytest -q` runs in under 1 minute on CPU.

---

## 11. Phases (implementation order)

1. Scaffold, config and synthetic fixtures + tests
2. Download + load + clean (50M)
3. **EDA** → decisions recorded in `reports/eda_summary.md`
4. Split, vocab, features, sequences
5. Metrics + evaluation protocol (tested)
6. MostPop / DecayPop / iALS baselines on 50M
7. SASRec baseline → reproduce the paper's ballpark on 50M (Kaggle GPU)
8. HookSeq additions one at a time → ablations on 50M
9. Scale to 500M on Kaggle (chunked preprocessing; fallback plan below)
10. Beyond-accuracy metrics, slices, bootstrap CIs, report generation
11. Gradio demo
12. README polish and final results

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 500M doesn't fit Kaggle RAM/disk | Use the `sequential/500m` format; Polars streaming; process per user shard; fallback: report on 50M + a 500M user subsample, stated clearly |
| Item vocab too large for GPU (≈3M items) | `min_count` vocab + ID-less audio path; d=64; sampled softmax (no full softmax in training) |
| Can't reproduce paper numbers | Debug the evaluation first with popularity baselines (cheap, deterministic) and compare against the paper's MostPop/DecayPop rows |
| Audio embeddings don't help warm items | Still report it; the value is in the cold slice. Ablation results are honest either way |
| Kaggle 12 h session limit | Checkpoint every epoch; resumable training; cached preprocessing artifacts |
| Mac sandbox lacks torch | Develop and test locally in a venv on the Mac (CPU/MPS); GPU runs on Kaggle |

---

## 13. Out of scope (YAGNI)

- A second-stage GBDT ranker (approach A). It can be mentioned as future work.
- Real-time serving, feature stores, online A/B testing (discussed in the README as "how this would ship at Hook").
- Bandit / exploration simulation.
- Raw audio processing (we use the provided embeddings).
