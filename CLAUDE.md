# HookSeq — project guardrails

**Purpose:** a portfolio project for the Hook Music *ML Engineering Intern, Recommendation Systems* application. Every decision should make the final README more convincing to Hook's hiring team.

**North star:** a SASRec-style next-track transformer on Yambda that **beats a faithfully reproduced SASRec** under the official protocol. It does so by using signals Hook actually has:
- audio embeddings, so new and newly licensed tracks can be recommended;
- event type (listen full / skip / like / dislike);
- organic vs. served (recommendation feedback loops).

It is backed by EDA-driven preprocessing, honest ablations, and artist-exposure metrics.

**Source of truth:** `docs/superpowers/specs/2026-11-12-hookseq-design.md` (spec), then `docs/superpowers/plans/2026-11-12-hookseq.md` (plan). The spec wins conflicts. Spec §13 lists what is out of scope.

## Drift checks (before adding anything)
1. Does it serve one of the 5 success criteria in spec §1? If not, don't build it. Note it under README "Future work".
2. **Training never runs locally.** The Mac runs code + `pytest -q` on synthetic fixtures only. All real runs follow `docs/kaggle_runbook.md`: CPU notebooks for download/EDA/preprocess/baselines, GPU only for transformer training and scoring, ~20 GPU-h/week budget, 9 h sessions with checkpoint resume.
3. **Evaluation stays comparable:** Yambda Global Temporal Split (300 d / 30 min gap / 1 d), full-catalog ranking, NDCG/Recall/Coverage @10/100, the paper's numbers as a reference row. Never switch to random splits or sampled-candidate metrics.
4. **EDA decides preprocessing:** every threshold or filter in `configs/data_*.yaml` must point to a finding in `reports/eda_summary.md`.
5. **Order matters:** reproduce the baselines and SASRec within ~15% of the paper *before* spending GPU on HookSeq variants.
6. **Honest results:** report ablations that don't help. No number goes in the README without a matching `results/*.json`.
7. **No scope creep:** no GBDT ranker, serving stack, bandits or raw-audio processing (spec §13) unless every success criterion is already met.

## Workflow
- Execute the plan task by task with tests first (superpowers subagent-driven development was chosen). Note: run Task 9 before Task 8.
- Commit after every task; record GPU hours actually used in the runbook.
