# HookSeq decision log

This log records every design decision that shaped a reported number, in roughly the order it
was taken. Each entry gives the decision, the alternatives that were on the table, the reason,
the evidence, and what it would cost if the decision turned out to be wrong. The
[README](../README.md) cites entries as D1, D2, ...

Numbers come from `results/*.json`, [`reports/results.md`](../reports/results.md) (generated from
`results/`), [`reports/eda_summary.md`](../reports/eda_summary.md) /
[`reports/eda_stats.json`](../reports/eda_stats.json), the [Kaggle runbook](kaggle_runbook.md) or
the Kaggle version logs. Where a decision rests on reading the official Yambda benchmark code,
the file is named; that code ships inside the Hugging Face dataset repo `yandex/yambda`
(`benchmarks/...`).

---

## Infrastructure and protocol

### D1. All training on Kaggle's free tier; the laptop only runs unit tests
- **Decision.** No model is trained locally. The laptop writes code and runs `pytest` on a
  synthetic Yambda-shaped fixture. Download, EDA, preprocessing and the CPU baselines run in CPU
  notebooks; only transformer training and 500M scoring use a GPU (one T4, fp16 AMP).
- **Alternatives.** Local training on a subsample; a paid cloud GPU.
- **Why.** A project constraint, and it forces every stage to be resumable and reproducible from
  a pinned tag (D22).
- **Evidence.** The design spec (§1, §9a); the measured budget in the runbook (§6): about 14.8
  GPU-hours in total against a planned 20.
- **Cost if wrong.** No effect on results; slower iteration.

### D2. Reproduce the official Global Temporal Split from the benchmark code, not from the paper text
- **Decision.** Test = the last day, `[t_test, ∞)` with `t_test = 26,000,000 − 86,400 = 25,913,600`
  s (the official constant); train = events before `t_test − 30 min` (the gap is carved from the
  train side). Validation mirrors it one day earlier: val = the last train day, val-train = events
  before `val_start − 30 min`. Timestamps are seconds.
- **Alternatives.** The first implementation followed the plan: test day measured forward from
  the end of 300 train days, gap carved from the test day, timestamps read as 5-second units.
- **Why.** Comparability with the paper is a hard requirement. Reading
  `benchmarks/yambda/processing/timesplit.py` and `constants.py` showed the split is anchored at
  the last timestamp with the gap on the train side, and that raw timestamps are seconds on a
  5-second grid, not 5-second units.
- **Evidence.** The 500M bundle's `meta.json` (K2 log): `t_train_end = 25,911,800`,
  `t_test_start = 25,913,600`, `t_test_end = 26,000,001`; the 30-minute gap is exactly 1,800 s.
  The popularity baselines, which have no learned parameters to hide a protocol error, then
  matched the paper within 7% (D5).
- **Cost if wrong.** Every number would be off-protocol; split rework and a full rerun.

### D3. Evaluate warm users; report cold users separately
- **Decision.** The headline metrics cover test users with at least one Listen+ train event (the
  official benchmark keeps only test users seen in train). Cold test users are scored with a
  MostPop fallback and reported in their own row.
- **Alternatives.** Include cold users in the headline; drop them silently.
- **Why.** The official protocol excludes them; the design review still wanted them handled.
- **Evidence.** 1 cold test user at 50M and 8 at 500M (`results/*-test-eval-*.json`,
  `n_cold_users`), so the choice cannot move the headline.
- **Cost if wrong.** Negligible at these counts.

### D4. Ground truth is Listen+ only, with a 50% played-ratio threshold
- **Decision.** A listen with `played_ratio_pct ≥ 50` is LISTEN_FULL (the paper's Listen+). The
  test truth of a user is the set of items with a Listen+ event on the test day. Likes are not
  truth. Warm users have at least one Listen+ train event.
- **Alternatives.** Truth = Listen+ or like (the first implementation); another threshold.
- **Why.** The paper's Table 6 is the Listen+ setting. The EDA shows a strongly bimodal played
  ratio, so the threshold barely matters near 50%.
- **Evidence.** EDA §2: the 95–100% bin holds 63.3% of served and 52.2% of organic listens, the
  0–5% bin 18.8% and 25.9%, and every middle bin at most 1–4%.
- **Cost if wrong.** One config value (`eval.target`); a rerun of evaluation.

### D5. Baselines fit on full-period aggregates; the DecayPop grid matches the official one
- **Decision.** MostPop, DecayPop and iALS fit on Listen+ aggregates of the entire training
  period computed during preprocessing, not on the stored (possibly truncated) sequences.
  DecayPop's half-life grid is the official per-window decay of 0.9 converted to half-lives
  (0.5 h to 24 h windows, i.e. 0.14–6.6 days) plus 30 days, with hourly bins for the last 30
  days so that sub-day half-lives are meaningful.
- **Alternatives.** Fit on the sequences (the first implementation); a day-binned grid.
- **Why.** The official baselines fit on all training rows. An internal review found the
  sequence-fitted version could not see old events of long histories.
- **Evidence.** After the change all three baselines land within 7% of the paper on NDCG@10 at
  both sizes (README §6.4): 50M MostPop 0.0177 vs 0.0186, DecayPop 0.0256 vs 0.0260,
  iALS 0.0379 vs 0.0407; 500M MostPop 0.0161 vs 0.0173, DecayPop 0.0263 vs 0.0267, iALS
  0.0392 vs 0.0384.
- **Cost if wrong.** Baseline rows only; cheap CPU reruns.

### D6. Report two recall definitions
- **Decision.** `recall@k` = hits@k / min(|T|, k), the definition in the official
  `metrics.py`; `recall_paper@k` = hits@k / |T|. Paper rows are compared against
  `recall_paper@k` only.
- **Alternatives.** Report only the official definition and accept a mismatch with the paper's
  Recall column (2.0–3.1× at k = 10, 1.12–1.46× at k = 100); report only the paper-compatible
  one.
- **Why.** Our NDCG agreed with the paper within 7%, while the official recall did not. The
  official repo has a later commit "Fix yet another bug in metrics (#24)" and a leftover comment
  suggesting the denominator used to be |T|, so the paper's table most likely predates that fix
  (a strong hypothesis, not verified).
- **Evidence.** A like-for-like pair, 500M MostPop at k = 10: `recall@10` 0.0153 and
  `recall_paper@10` 0.0068, against the paper's 0.0060. hits/|T| is much closer but not equal.
  Where both are available (the 500M baselines and 50M `sasrec_bce`), `recall_paper@k` is
  12–38% above the paper at k ∈ {10, 100}, while NDCG@10 is within 7%. That residual gap is
  unexplained (`reports/results.md`).
- **Cost if wrong.** An extra column.

## Baselines and training

### D7. Two SASRec rows, plus a same-code-path control
- **Decision.** (i) `sasrec_bce`: the paper-faithful reproduction (official architecture: post-LN,
  LayerNorm on the input embeddings, no √d scaling, no final LayerNorm; BCE with one uniform
  negative per position drawn from the Listen+ catalog; Adam with a constant 1e-3, no dropout,
  no clipping, fp32; a fixed 3,500 steps ≈ 100 epochs; scoring masked to the Listen+ catalog the
  official evaluation ranks). (ii) `sasrec`: a strong SASRec (pre-LN, sampled softmax with 4,096
  in-batch plus 8,192 uniform negatives, no logQ, AdamW with warmup + cosine, dropout 0.2, early
  stopping). This is the baseline HookSeq must beat. (iii) `a0_sasrec`: HookSeq's own code path
  with every addition off, the base of the ablation ladder.
- **Alternatives.** One SASRec row only.
- **Why.** Success criterion 1 asks for a faithful reproduction, and criterion 2 asks HookSeq to
  beat SASRec. Beating only the weak BCE row would be an easy and unconvincing bar; the strong
  row is the honest one. The a0 control isolates the ablation steps from the input-setting
  differences of D8.
- **Evidence.** 50M test NDCG@10: `sasrec_bce` 0.0771 (paper 0.0748, +3.1%), `sasrec` 0.0970,
  `a0_sasrec` 0.1074.
- **Cost if wrong.** One extra 50M run (0.21 GPU-h).

### D8. Input setting per row: Listen+ last window for `sasrec`, every event and random windows for HookSeq
- **Decision.** `sasrec` and `sasrec_bce` read Listen+ events only and train on one last window
  of 201 events per user (as the official SASRec). HookSeq and every ablation read all event
  types (skips, likes, dislikes as context) and sample a random window per user and epoch,
  weighted towards recent windows. Targets are positive events (Listen+ or like) in both cases.
- **Alternatives.** Give every row the same input.
- **Why.** The `sasrec` row must stay paper-comparable, while HookSeq's whole point is to use the
  signals SASRec ignores. The a0 control (D7) carries HookSeq's input setting, so the ablation
  ladder compares like with like.
- **Evidence.** a0 (0.1074) vs `sasrec` (0.0970) at 50M: the input setting, ID table and ID dropout together,
  without any HookSeq feature, move NDCG@10 by +0.0104 (unpaired). The four factors are not separated (see README
  §9).
- **Cost if wrong.** One more config.

### D9. Two-pass training: early-stop on validation, refit on the full train period
- **Decision.** A val pass trains on val-train, evaluates val NDCG@10 every 500 steps, stops after
  6 evaluations without improvement and records the best step. The test pass retrains from
  scratch on the full training period for exactly that many steps, with the same LR schedule
  horizon, and is evaluated once. The exception is `sasrec_bce`: like the official recipe, it
  trains a fixed 3,500 steps in both passes and is evaluated once at the end (D7).
- **Alternatives.** Evaluate the val-trained model on test (it misses the last train day).
- **Why.** It mirrors the official hyperparameter-search and final passes, and the test model sees
  the day just before the test day.
- **Evidence.** Every neural `*-test-*.json` has `stopped_reason = max_steps` and `steps` equal to
  its val record's `best_step` (for example 500M `sasrec`: 33,000; 500M a8: 20,000).
- **Cost if wrong.** A test model slightly under- or over-trained.

### D10. Negatives: 4,096 in-batch (with replacement) + 8,192 uniform, logQ from expected counts
- **Decision.** Each step shares one negative set: 4,096 items drawn with replacement from the
  batch's positive targets (popularity-proportional) plus 8,192 uniform catalog items. With
  logQ on, logits are corrected by log of each item's expected count in that set, using
  positive-event popularity of the training period actually used (val-train for val, train for
  test). Accidental hits are masked.
- **Alternatives.** Full in-batch negatives (an M × (M + 8,192) logit block, with M up to about
  51K valid positions per batch, does not fit a T4); uniform only.
- **Why.** Mixed negatives are standard for large catalogues; logQ removes the popularity bias the
  in-batch half introduces.
- **Evidence.** logQ is the largest single ablation gain (README §6.6).
- **Cost if wrong.** The size of the logQ effect.

### D11. A chunked, fp16 sampled-softmax kernel, after profiling
- **Decision.** The sampled softmax is a custom autograd function that never materialises the
  full [M, S + 1] logit matrix: rows go in chunks, matmuls run in fp16 on tensor cores, softmax
  statistics accumulate in fp32, and gradients are computed in the forward pass. The result is
  numerically the same loss.
- **Alternatives.** Fewer negatives; a row subsample per batch.
- **Why.** The first SASRec run trained far too slowly to afford the ablation ladder and 500M.
  A dedicated profiling notebook showed the fp32 loss and its backward pass dominated the step;
  the data loader and encoder were not the bottleneck. Shrinking the negative set would have
  changed the method.
- **Evidence.** After the change: 50M `sasrec` 6.0 steps/s and the HookSeq-path runs 7.0–9.4
  steps/s on one T4 (`steps_per_sec` in the records). The pre-change rates and the profile are
  not kept in `results/` and are not quoted.
- **Cost if wrong.** A subtle numerical bug; covered by unit tests against a reference
  implementation.

### D12. Dislikes as hard negatives, window-level
- **Decision.** With `loss.dislike_hard_negatives`, each sequence row adds up to its 4 most recent
  disliked items from the input window as extra negatives for all of its target positions.
- **Alternatives.** Position-causal dislikes (only those before each position); ignore dislikes.
- **Why.** Cheap, and a disliked item is a safe negative regardless of position.
- **Evidence.** a3 → a4: +0.0012 NDCG@10 at 50M, inside the confidence intervals (README §6.6).
- **Cost if wrong.** A small effect either way.

## Data and features

### D13. Preprocessing thresholds set by the EDA
- **Decision.** `min_count: 5` positive train events for an ID embedding; `session_gap_seconds:
  1800`; no collapsing of consecutive duplicates; model window `max_len: 200`; stored histories
  uncapped at 50M (`max_len_store: null`) and capped at 1,024 events at 500M; no filtering of
  already-seen items at evaluation.
- **Alternatives.** The automatic session-gap suggestion (15.9 s); collapsing duplicates;
  filtering seen items.
- **Why and evidence (EDA sections).** §1: the top 1% of items take 49.2% of events, the item
  Gini is 0.907, and roughly 3×10⁵ of 886K items have ≥ 5 events, so a cut-off of 5 keeps an ID
  for the items behind about 95–99% of traffic with a third of the table. §3: the 15.9 s
  "valley" is an artefact of the 5-second timestamp grid, so it was rejected. §4: re-listens are
  71.1% of listens but consecutive duplicates only 2.0%, so collapsing would remove little and
  seen-item filtering would remove legitimate targets. §7: the median history is 2,594 events
  and the 99th percentile 23,752. With the 1,024 cap, 500M preprocessing peaked at 13,091 MB
  RSS (K2 log), inside the 14 GB session; whether it would have fit uncapped was not tested.
- **Cost if wrong.** `min_count` is the most consequential; D18 tests it directly.

### D14. The item catalogue contains every split's items; unknown test items stay as misses
- **Decision.** The vocabulary covers all items of the subset, so items first seen on the test
  day have an index and audio/artist features. Items without enough train positives get no ID
  row. Nothing is dropped from the truth.
- **Alternatives.** A train-only catalogue (test-only items would be out of vocabulary).
- **Why.** Content features can only score a new track if it is in the catalogue being ranked.
- **Evidence.** Distinct test-truth items (Listen+ items of warm users) with no train event:
  2.2% at 50M and 1.3% at 500M (`new_items_share.meta` in the eval records). This is lower than
  the EDA's 3.0% (EDA §10), which counts distinct test-day items of any event type.
- **Cost if wrong.** None for warm metrics.

### D15. The item tower: an additive ID + audio + artist representation with ID dropout
- **Decision.** `item_repr = ID_emb · keep + MLP(audio) + artist_emb`, where `keep` drops the ID
  with probability 0.2 in training (no rescaling, so training and inference see the same scale),
  items without audio use a learned missing-audio vector, audio vectors are PCA-reduced to 128
  dimensions (fit on items with a train event), and the artist is the primary artist. The same
  representation serves inputs, targets and negatives, and scores the whole catalogue.
- **Alternatives.** Concatenation and projection; a gating network; separate input and output
  tables.
- **Why.** The simplest design in which an item without an ID still has a representation in the
  same space, and with every flag off it reduces exactly to the SASRec parameterisation (a unit
  test checks parameter equality).
- **Evidence.** On the synthetic fixture a content-only model scored cold items while a
  shuffled-audio control scored 0 (unit test). On real data see README §7.3.
- **Cost if wrong.** The cold-start mechanism is weak (it was; README §8).

## Ablations and slices

### D16. A cumulative ablation ladder
- **Decision.** a0 (all off) → a1 (+audio, +artist) → a2 (+event type, organic flag, played-ratio
  bucket inputs) → a3 (+logQ) → a4 (+dislike hard negatives = `configs/hookseq.yaml`). Two
  stretch steps (a5 like head, a6 time-gap embedding) were configured but not run.
- **Alternatives.** A full factorial design (2⁵ runs) or leave-one-out from the full model.
- **Why.** The GPU budget allowed about five 50M configurations; a ladder answers "what does each
  addition contribute on top of the previous ones".
- **Evidence.** One session of 3.63 h wall time, 3.58 GPU-h (runbook §6).
- **Cost if wrong.** Order effects are not identified: a feature that hurts early may help later
  (D18 found exactly this for content features).

### D17. Split the "cold item" slice into `idless_items` and `new_items`
- **Decision.** `idless_items` = truth items with no ID in the data vocabulary (fewer than 5 train
  positives); `new_items` = truth items with no train event of any kind, a subset that only
  content features can score. Both are defined on the data vocabulary, so they are the same items
  for every model.
- **Alternatives.** The original single `cold_items` slice.
- **Why.** The first 50M report showed SASRec (NDCG@10 0.0184) beating the audio models a1–a4
  (0.0061–0.0098) on `cold_items`. The slice turned out to mean "no HookSeq ID", and SASRec has an ID row for every
  catalogue item, so it was not a cold-start slice at all. The EDA had promised a never-seen
  slice, and it was missing.
- **Evidence.** `reports/results.md`, slices of 50M `sasrec` and a1–a4 (`idless_items`).
- **Cost if wrong.** Mislabelled cold-start claims; it was caught before any claim was written.

### D18. Two follow-up controls: a7 (no content) and a8 (an ID for every item)
- **Decision.** a7 = the full model without audio and artist. a8 = the full model with
  `model.id_min_count: 1`, so every item with at least one train positive gets an ID and content
  adds to it instead of replacing it. a8 rebuilds the ID table at model-build time from the same
  bundle, so every 50M run shares one processed directory.
- **Alternatives.** Re-preprocess with `min_count: 1` (a second bundle; cross-bundle comparisons
  would not be paired).
- **Why.** a1 hurt warm accuracy (−0.0107 vs a0). Two explanations were possible: content
  features are harmful, or the cost is the `min_count` threshold (items with 1–4 positives lose
  their ID and must live on content alone). a7 tests the first, a8 the second.
- **Evidence.** a7 is the best 50M model on val (0.1215) and test (0.1207); a8 0.1184 (val
  0.1168). Content features do not add warm accuracy to the full model. README §6.7.
- **Cost if wrong.** 1.83 GPU-h.

### D19. The 500M HookSeq run is a8, not the validation-best a7
- **Decision.** Train a8 at 500M.
- **Alternatives.** a7 (best on 50M validation); a4 (the nominal `hookseq.yaml`); both a7 and a8.
- **Why.** The budget allowed one 500M HookSeq run. a7 has no content path, so it cannot score a
  never-seen track at all, which is the Hook motivation for the project; a8 keeps the content
  path and beat `sasrec` by +0.0214 NDCG@10 at 50M (paired CI [+0.0175, +0.0252]). a8 and a4
  were tied on validation (0.1168 vs 0.1166); a8 was higher on test (0.1184 vs 0.1163) and on
  `idless_items` (0.0146 vs 0.0066), but lower on `new_items` (0.0012 vs 0.0039), the slice
  the content path exists for.
- **Evidence.** 50M records and `reports/results.md`.
- **Cost if wrong.** The 500M headline may understate the best achievable model by up to the a7–a8
  gap (0.0023 at 50M); an a7 500M run costs about 3–5 GPU-h. This choice also used 50M *test*
  numbers, which is a mild form of test reuse (README §9).

### D20. A smaller evaluation batch for 500M HookSeq
- **Decision.** `train.eval_batch_size=256` for the 500M a8 run (operational key, not in the
  training hash).
- **Alternatives.** Evaluate less often; subsample validation users.
- **Why.** The first 500M a8 version crashed with a CUDA out-of-memory error at its first periodic
  evaluation: the process already held 11.47 GiB of the T4's 14.56 GiB, and the top-k tie check
  asked for 3.82 GiB more (K4 log).
- **Evidence.** The second version completed (K4 log: val 0.1018, test 0.1009, 3.15 GPU-h).
- **Cost if wrong.** None on results; the failed attempt cost about 0.12 h of session time.

### D21. Paired bootstrap over users, against the strong SASRec first
- **Decision.** The headline comparison is a paired bootstrap (1,000 resamples, 95% percentile
  interval) of the per-user difference HookSeq − SASRec over the users both runs evaluate, with
  `sasrec` as the first baseline and `a0_sasrec` as the second. a7 and a8 get their own paired
  rows when they exist.
- **Alternatives.** Unpaired CIs only; a0 as the reference.
- **Why.** Paired differences remove between-user variance, which dominates per-user NDCG.
- **Evidence.** 500M: unpaired CIs of a8 and `sasrec` are [0.0992, 0.1026] and [0.0784, 0.0814];
  the paired difference is +0.0211 [+0.0198, +0.0223].
- **Cost if wrong.** None.

## Engineering and reporting

### D22. Reproducibility: hashed bundles, pinned tags, a git SHA per record
- **Decision.** A pipeline version is hashed into the processed-bundle name, and loading refuses
  bundles from another version; every reported run uses a notebook pinned to a git tag; every
  record stores the config, its hash and the git SHA.
- **Why.** Several protocol fixes (D2–D5) changed the data contract during the project, and stale
  bundles must not mix with new ones.
- **Evidence.** Records carry SHAs of the tags `v0.2-r2` (50M baselines), `v0.3` (`sasrec`,
  `sasrec_bce`), `v0.3.1` (a0–a4) and `v0.4` (a7, a8, all 500M runs, every evaluation record).
- **Cost if wrong.** None.

### D23. Notebooks for an unattended free tier
- **Decision.** Headless "import and run" notebooks that find their inputs by marker files,
  resume from earlier outputs, train a list of configs per session with a budget guard
  (`max_hours: 8.0`, no new split with under 40 minutes left), and at 500M allow at most two
  configs per session (disk) and refuse to start if the projected disk use is above 18 GB.
- **Why.** Kaggle sessions die at 9 h with 20 GB of working disk; a 500M checkpoint with optimizer
  state is large.
- **Evidence.** No reported run was lost to a session limit; 500M sessions ended with 1.83 GB
  (a8) and 2.18 GB (`sasrec`) of working disk (K4 logs).
- **Cost if wrong.** Engineering time only.

### D24. No hyperparameter sweep
- **Decision.** The planned four-run sweep of the best configuration was not run; its budget went
  to the a7/a8 controls. All transformer rows share d = 64, 2 layers, 2 heads, batch 256, LR
  1e-3 (the official SASRec defaults).
- **Why.** The controls answered a question the ablation ladder had raised (D18); a sweep would
  only have raised every number.
- **Cost if wrong.** Absolute numbers are probably not optimal for any row, including `sasrec`.

### D25. The demo runs on Hugging Face ZeroGPU from a bundle stored in the Space
- **Decision.** A Gradio app with two tabs: session → next tracks (HookSeq next to SASRec) and
  new-track audience. It reads a self-contained bundle exported from the 500M test checkpoints
  of a8 and `sasrec` (56,120 items, 200 sample users, 57.1 MB), stored in the Space's `bundle/`
  directory, so there is no separate model repo. The code is pinned to tag `v0.1-demo` (= `v0.4`).
  Live at https://huggingface.co/spaces/Arush1050/musicreccmodel.
- **Alternatives.** A separate model repo for the bundle (supported via `HOOKSEQ_BUNDLE_REPO`);
  a CPU Space.
- **Why.** Free Hugging Face accounts get ZeroGPU hardware, and a 57 MB bundle fits in the
  Space repo.
- **Evidence.** Unit tests on a synthetic bundle; the Kaggle export log; a `gradio_client` call
  on 2026-11-14 (a sample user loaded with 20 session rows, then the HookSeq and SASRec top-5
  tables). See [docs/hf_space.md](hf_space.md).
  The new-track tab demonstrates the mechanism only: on the 500M `new_items` slice the model's
  NDCG@10 is 0.0003.
- **Cost if wrong.** The demo falls back to CPU.

### D26. The README is written as a short paper with a decision log
- **Decision.** Abstract, introduction, related work, data, protocol, method, a chronological
  account of the experiments, results, discussion, threats to validity, and this log.
- **Why.** The reader should be able to see what was tried, what failed, and why each number is
  what it is.
- **Cost if wrong.** A cheap restructure.
