"""Markdown results tables from run records (`save_run` JSONs) and the paper reference numbers."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger(__name__)

PAPER_REFERENCE = Path(__file__).resolve().parents[3] / "configs" / "paper_reference.yaml"
DEFAULT_METRICS = (
    "ndcg@10", "ndcg@100", "recall@10", "recall@100",
    "recall_paper@10", "recall_paper@100", "coverage@10",
)  # fmt: skip
# The paper's "Recall" column is hits/|T|, i.e. our recall_paper@k, not our official recall@k.
PAPER_RECALL_MAP = {"recall@10": "recall_paper@10", "recall@100": "recall_paper@100"}
RECALL_FOOTNOTE = (
    "_Recall definitions._ `recall@k` is the official Yambda definition: hits@k / min(|T|, k) "
    "(benchmarks/yambda/evaluation/metrics.py, after the bug-fix commit "
    '"Fix yet another bug in metrics (#24)", whose leftover comment suggests the denominator '
    "used to be |T|). `recall_paper@k` is hits@k / |T|. The paper's Recall column is much "
    "closer to hits/|T| than to min(|T|, k), though not equal to it: our NDCG@10 agrees with "
    "the paper within 7%, our min(|T|, k) recall is 2.0-3.1x the paper's at k=10 (1.12-1.46x at "
    "k=100), and hits/|T| is 12-38% above it. So paper rows are shown under `recall_paper@k` "
    "and their `recall@k` is left as "
    '"-"; compare paper Recall against `recall_paper@k` only. Older run records without '
    '`recall_paper@k` show "-".'
)
NOT_RUN = "not yet run"


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) and not math.isnan(x) else "-"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join([_row(header), "|" + "---|" * len(header), *(_row(r) for r in rows)])


_REPLICATE = re.compile(r"(.+)_s(\d+)")


def _replicate(name: str) -> tuple[str, int] | None:
    """(base, N) when `name` (no `-test` / `-test-eval` suffix) is seed replicate N >= 2."""
    m = _REPLICATE.fullmatch(name)
    return (m.group(1), int(m.group(2))) if m and int(m.group(2)) >= 2 else None


def _is_replicate_run(run_name: str) -> bool:
    return _replicate(run_name.removesuffix("-eval").removesuffix("-test")) is not None


def _is_eval(r: dict) -> bool:
    return r["run_name"].endswith("-eval") or r.get("extra", {}).get("kind") == "eval"


def _latest(
    runs: list[dict], split: str, size: str | None, evals: bool = False
) -> dict[tuple[str, str | None], dict]:
    """Latest record per (run_name, size) for `split` (and dataset size, if given).

    `--quick` records (`extra.quick`, a one-point smoke grid) never count.
    """
    out: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: r["timestamp"]):
        ex = r.get("extra", {})
        if _is_eval(r) != evals or ex.get("split") != split or ex.get("quick"):
            continue
        if size is not None and ex.get("dataset_size") != size:
            continue
        out[(r["run_name"], ex.get("dataset_size"))] = r
    return out


def paper_rows(size: str, path: Path = PAPER_REFERENCE) -> dict[str, dict]:
    """Paper reference numbers for `size` (model -> metric -> value); empty if unavailable."""
    if not Path(path).exists():
        return {}
    ref = yaml.safe_load(Path(path).read_text()) or {}
    out = {}
    for model, vals in (ref.get(str(size).lower()) or {}).items():
        # the paper's recall is hits/|T|: file it under recall_paper@k, never under recall@k
        out[model] = {PAPER_RECALL_MAP.get(m, m): v for m, v in (vals or {}).items()}
    return out


def results_table(
    runs: list[dict],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    split: str = "test",
    size: str = "50m",
) -> str:
    """Latest run per name for `split` and `size`, then the paper rows marked "(paper)".

    Every row is in the paper's Listen+ setting (its Table 6): Listen+ ground truth over warm
    users (>= 1 Listen+ train event), baselines fit on the full train period.
    """
    rows = []
    for (name, _), r in sorted(
        _latest(runs, split, size).items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
    ):
        if _is_replicate_run(name):  # seed replicates are summarized in their own section
            continue
        label = name.removesuffix(f"-{split}")
        rows.append([label, *(_fmt(r["metrics"].get(m)) for m in metrics)])
    if not rows:
        rows.append([f"{NOT_RUN} ({size}, {split})", *["-"] * len(metrics)])
    for model, vals in paper_rows(size).items():
        rows.append([f"{model} (paper)", *(_fmt(vals.get(m)) for m in metrics)])
    return _table(["model", *metrics], rows)


def ablation_table(
    runs: list[dict], prefix: str = "a", split: str = "test", size: str | None = None
) -> str:
    """Ablation runs (`^a\\d+_`) ordered a0, a1, ...; delta vs a0 for ndcg@10 and recall@100."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)_")
    found = {}
    for (name, _), r in _latest(runs, split, size).items():
        m = None if _is_replicate_run(name) else pat.match(name)
        if m:
            found[int(m.group(1))] = (name, r)
    header = ["run", "ndcg@10", "d ndcg@10", "recall@100", "d recall@100"]
    if not found:
        return _table(header, [[NOT_RUN, "-", "-", "-", "-"]])
    base = found[0][1]["metrics"] if 0 in found else None
    rows = []
    for i in sorted(found):
        name, r = found[i]
        cells = [name.removesuffix(f"-{split}")]
        for m in ("ndcg@10", "recall@100"):
            v = r["metrics"].get(m)
            d = None if base is None or v is None or base.get(m) is None else v - base[m]
            cells += [_fmt(v), "n/a" if d is None else ("-" if i == 0 else f"{d:+.4f}")]
        rows.append(cells)
    return _table(header, rows)


SLICE_KEYS = ("ndcg@10", "recall@10", "ndcg@100", "recall@100", "coverage@10")
# eval records written before F5 call idless_items "cold_items" (the same definition)
LEGACY_SLICE_NAMES = {"cold_items": "idless_items"}
SLICE_FOOTNOTE = (
    "_Slices._ `organic` / `served`: truth items with an organic / recommender-served test "
    "event. `idless_items`: truth items without an ID in the data vocab (fewer than "
    "`data.min_count` train positives) or out of vocabulary; the same items for every model, "
    "whatever its own ID table (`model.id_min_count`); records before F5 named it "
    "`cold_items`. `new_items`: truth items with no train-period event of any kind (first "
    "seen on the test day), a subset of `idless_items`; only content features can score them. "
    "`warm_items`: truth items with a data-vocab ID. `users_low/mid/high`: terciles of the "
    "users' Listen+ train events."
)


def slice_table(eval_run: dict | None) -> str:
    """Per-slice metrics of one eval record (`extra.slices`)."""
    header = [
        "slice",
        "users",
        "ndcg@10",
        "recall@10",
        "ndcg@100",
        "recall@100",
        "coverage@10 (within slice)",
    ]
    sl = (eval_run or {}).get("extra", {}).get("slices")
    if not sl:
        return _table(header, [[NOT_RUN, "-", "-", "-", "-", "-", "-"]])
    rows = []
    for name, m in sl.items():
        label = LEGACY_SLICE_NAMES.get(name, name)
        rows.append([label, int(m.get("n_users", 0)), *(_fmt(m.get(c)) for c in SLICE_KEYS)])
    cold = eval_run["extra"].get("metrics_cold") or {}
    if cold:
        rows.append(
            [f"cold users (n={eval_run['extra'].get('n_cold_users', '?')})", "-"]
            + [_fmt(cold.get(c)) for c in SLICE_KEYS]
        )
    return _table(header, rows)


FAIRNESS_COLS = (
    "artist_gini@10", "artist_coverage@10", "long_tail_share@10",
    "artist_gini@100", "artist_coverage@100", "long_tail_share@100",
)  # fmt: skip


def _fmt_ci(c) -> str:
    return f"{c[0]:.4f} [{c[1]:.4f}, {c[2]:.4f}]" if c else "-"


def fairness_table(eval_runs: list[dict]) -> str:
    """Exposure fairness (and bootstrap CIs) per eval record."""
    header = ["run", *FAIRNESS_COLS, "ndcg@10 [95% CI]", "recall@100 [95% CI]"]
    rows = []
    for r in eval_runs:
        ex = r.get("extra", {})
        fair = ex.get("fairness") or {}
        ci = ex.get("ci") or {}

        label = r["run_name"].removesuffix("-eval")
        rows.append([label, *(_fmt(fair.get(c)) for c in FAIRNESS_COLS), _fmt_ci(ci.get("ndcg@10")),
                     _fmt_ci(ci.get("recall@100"))])  # fmt: skip
    if not rows:
        rows.append([NOT_RUN, *["-"] * (len(header) - 1)])
    return _table(header, rows)


def paired_diff_ci(
    npz_a, npz_b, metric: str, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a - b) over the users common to both per-user files.

    Returns (mean, lo, hi) of the 95% percentile interval; NaNs if no user is shared.
    """
    with np.load(npz_a) as a, np.load(npz_b) as b:
        ua, ub = a["uids"], b["uids"]
        common, ia, ib = np.intersect1d(ua, ub, return_indices=True)
        diff = a[metric][ia].astype(np.float64) - b[metric][ib].astype(np.float64)
    if len(common) == 0:
        return (float("nan"),) * 3
    from hookrec.eval.bootstrap import bootstrap_ci

    return bootstrap_ci(diff, n_boot=n_boot, seed=seed)


# Paired-diff baselines, preferred first: the paper-comparable SASRec row, then the same-code-path
# control (HookSeq with every flag off, all-event input).
PAIRED_BASELINES = (
    ("sasrec", "strong SASRec, Listen+ input"),
    ("a0_sasrec", "a0 control, same code path"),
)


# Runs that stand in for HookSeq in the paired section, preferred first: a run named `hookseq`,
# then a4_full_dislike (its config equals configs/hookseq.yaml), then the best HookSeq-model
# eval by test ndcg@10 (see `_hookseq_run`).
HOOKSEQ_RUNS = (
    ("hookseq", "the run named hookseq"),
    ("a4_full_dislike", "no run named hookseq; a4_full_dislike's config equals hookseq.yaml"),
)


def _hookseq_run(by: dict[str, dict]) -> tuple[str, str, dict] | None:
    """(run name, why it was chosen, eval record) of the run standing in for HookSeq.

    `by` maps eval run names (`{run}-test-eval`) to records of one size. Without a `hookseq`
    or `a4_full_dislike` eval, the best test ndcg@10 among the evals whose `config.model.name`
    is hookseq, the paired baselines excluded (a0_sasrec is a HookSeq model with every flag
    off); ties break by name.
    """
    for name, why in HOOKSEQ_RUNS:
        if f"{name}-test-eval" in by:
            return name, why, by[f"{name}-test-eval"]
    skip = {n for n, _ in PAIRED_BASELINES}
    cands = []
    for run_name, r in by.items():
        name = run_name.removesuffix("-test-eval")
        model = ((r.get("config") or {}).get("model") or {}).get("name")
        if name == run_name or name in skip or model != "hookseq":
            continue
        v = (r.get("metrics") or {}).get("ndcg@10")
        if isinstance(v, (int, float)) and not math.isnan(v):
            cands.append((v, name, r))
    if not cands:
        return None
    _, name, r = max(cands, key=lambda c: (c[0], c[1]))
    if len(cands) == 1:
        why = "the only HookSeq-model run at this size"
    else:
        why = "no hookseq or a4_full_dislike run; best test ndcg@10 among HookSeq-model runs"
    return name, why, r


def _peruser_file(results_dir: Path, rec: dict, name: str, size: str) -> Path:
    """The eval record's per-user npz (`extra.peruser_file`, else run_eval's naming)."""
    fname = (rec.get("extra") or {}).get("peruser_file") or f"{name}-test-{size}-eval-peruser.npz"
    return results_dir / fname


# Follow-up controls that get their own paired rows (after the HookSeq stand-in's) whenever
# their eval record and per-user npz exist for the size: a7 = full model minus audio/artist,
# a8 = full model with an ID for every item (F5). They separate "content features help" from
# "the ID-vocab threshold" in the paired comparison.
EXTRA_PAIRED_RUNS = ("a7_full_no_content", "a8_full_all_ids")


def _paired_rows(
    a_name: str, a_rec: dict, label: str, found: list, results_dir: Path, size: str
) -> list[list]:
    """Rows of `a_name` minus each baseline in `found`, labelled `label` in column one."""
    a = _peruser_file(results_dir, a_rec, a_name, size)
    rows = []
    for name, base_label, b_rec in found:
        base = f"{label} vs {name} ({base_label})"
        if a_rec["extra"].get("processed_dir") != b_rec["extra"].get("processed_dir"):
            rows.append([base, "not available (processed_dir mismatch)", "-", "-"])
            continue
        b = _peruser_file(results_dir, b_rec, name, size)
        for m in ("ndcg@10", "recall@100"):
            try:
                mean, lo, hi = paired_diff_ci(a, b, m)
            except Exception as e:  # noqa: BLE001  # missing / corrupt npz
                log.warning("paired diff %s vs %s unavailable: %s", m, name, e)
                rows.append([base, f"{m}: not available", "-", "-"])
                continue
            diff = [f"{m} ({a_name} - {name})", f"{mean:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"]
            rows.append([base, *diff])
    return rows


def _paired_section(evals: list[dict], results_dir: Path, size: str) -> str:
    """HookSeq vs SASRec paired diffs for one size, against every evaluated baseline of
    `PAIRED_BASELINES` (`sasrec` first); each needs the same processed_dir as HookSeq.

    The HookSeq side is `_hookseq_run`'s choice, named in every row and in a caption line.
    Each run of `EXTRA_PAIRED_RUNS` other than that choice follows with its own rows, if its
    eval record and per-user npz exist for `size`."""
    header = ["comparison", "metric", "mean diff", "95% CI"]
    by = {r["run_name"]: r for r in evals if r.get("extra", {}).get("dataset_size") == size}
    hs = _hookseq_run(by)
    found = [
        (n, label, by[f"{n}-test-eval"]) for n, label in PAIRED_BASELINES if f"{n}-test-eval" in by
    ]
    if hs is None or not found:
        return _table(header, [[f"HookSeq vs SASRec: {NOT_RUN}", "-", "-", "-"]])
    hs_name, why, a_rec = hs
    rows = _paired_rows(hs_name, a_rec, f"HookSeq ({hs_name})", found, results_dir, size)
    for name in EXTRA_PAIRED_RUNS:
        rec = by.get(f"{name}-test-eval")
        if name == hs_name or rec is None:
            continue
        if not _peruser_file(results_dir, rec, name, size).is_file():
            continue
        rows += _paired_rows(name, rec, name, found, results_dir, size)
    caption = f"_HookSeq = `{hs_name}` ({why})._"
    return caption + "\n\n" + _table(header, rows)


SEED_METRICS = ("ndcg@10", "recall@100", "coverage@10", "long_tail_share@10")
SEED_CAPTION = (
    "_Replicate 1 is the un-suffixed run (`train.seed: 42`); `<base>_sN` is replicate N. Values "
    "come from the eval records (`-test-eval`: metrics, and `long_tail_share@10` from the "
    "fairness block), falling back to the training test records when a run has no eval record "
    "(the headline table above uses the training test records). Cells are mean ± sample std, "
    "then the per-seed values in seed order._"
)


def _val(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not math.isnan(x) else None


def _mean_std(vals: list[float]) -> tuple[float, float | None]:
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    return float(np.mean(vals)), sd


def _seed_metric(test: dict | None, ev: dict | None, metric: str) -> float | None:
    for rec in (ev, test):
        if rec is None:
            continue
        v = _val((rec.get("metrics") or {}).get(metric))
        if v is None:
            v = _val(((rec.get("extra") or {}).get("fairness") or {}).get(metric))
        if v is not None:
            return v
    return None


def _seed_groups(runs: list[dict], size: str) -> dict[str, dict[int, dict]]:
    """base -> seed N -> {"test": record, "eval": record, "name": run name}, for bases with
    at least one replicate N >= 2 (the un-suffixed run is replicate 1)."""
    groups: dict[str, dict[int, dict]] = {}
    for kind, evals in (("test", False), ("eval", True)):
        for (name, _), r in _latest(runs, "test", size, evals=evals).items():
            name = name.removesuffix("-eval").removesuffix("-test")
            base, n = _replicate(name) or (name, 1)
            groups.setdefault(base, {}).setdefault(n, {"name": name})[kind] = r
    return {b: g for b, g in groups.items() if any(n >= 2 for n in g)}


def _spread(vals: list[float], signed: bool = False) -> str:
    """mean ± sample std (just the mean for a single value)."""
    if not vals:
        return "-"
    mean, sd = _mean_std(vals)
    if signed:
        return f"{mean:+.4f}" if sd is None else f"{mean:+.4f} ± {sd:.4f}"
    return _fmt(mean) if sd is None else f"{_fmt(mean)} ± {_fmt(sd)}"


def _seed_rows(groups: dict[str, dict[int, dict]]) -> list[list]:
    rows = []
    for base in sorted(groups):
        g = groups[base]
        cells = [base, len(g)]
        for m in SEED_METRICS:
            per = [_seed_metric(g[n].get("test"), g[n].get("eval"), m) for n in sorted(g)]
            vals = [v for v in per if v is not None]
            cells.append(f"{_spread(vals)} ({', '.join(_fmt(v) for v in per)})" if vals else "-")
        rows.append(cells)
    return rows


def _seed_paired_rows(
    groups: dict[str, dict[int, dict]], results_dir: Path, size: str
) -> tuple[list[list], list[str]]:
    """Per-seed paired diffs (HookSeq-side replicated base minus each replicated baseline of
    `PAIRED_BASELINES`, matched by seed N), a summary row per metric, and notes for skips."""
    rows, notes = [], []
    baselines = [(n, lbl) for n, lbl in PAIRED_BASELINES if n in groups]
    for base in sorted(b for b in groups if b not in {n for n, _ in PAIRED_BASELINES}):
        for bname, _ in baselines:
            label = f"{base} - {bname}"
            diffs: dict[str, list[tuple[float, float, float]]] = {m: [] for m in PAIRED_METRICS}
            for n in sorted(set(groups[base]) & set(groups[bname])):
                a, b = groups[base][n].get("eval"), groups[bname][n].get("eval")
                if a is None or b is None:
                    notes.append(f"{label}, seed {n}: no eval record for one side, skipped")
                    continue
                if a["extra"].get("processed_dir") != b["extra"].get("processed_dir"):
                    notes.append(f"{label}, seed {n}: processed_dir mismatch, skipped")
                    continue
                fa = _peruser_file(results_dir, a, groups[base][n]["name"], size)
                fb = _peruser_file(results_dir, b, groups[bname][n]["name"], size)
                for m in PAIRED_METRICS:
                    try:
                        mean, lo, hi = paired_diff_ci(fa, fb, m)
                    except Exception as e:  # noqa: BLE001  # missing / corrupt npz
                        log.warning("seed paired diff %s seed %d unavailable: %s", label, n, e)
                        notes.append(
                            f"{label}, seed {n}: {m} not available (per-user file), skipped"
                        )
                        continue
                    diffs[m].append((mean, lo, hi))
                    rows.append([label, f"seed {n}", m, f"{mean:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
            for m, d in diffs.items():
                if d:
                    summ = f"{_spread([x[0] for x in d], signed=True)} (n={len(d)})"
                    bounds = f"[{min(x[1] for x in d):+.4f}, {max(x[2] for x in d):+.4f}]"
                    rows.append([label, "summary", m, summ, f"min lo / max hi {bounds}"])
    return rows, notes


PAIRED_METRICS = ("ndcg@10", "recall@100")


def _seed_section(runs: list[dict], results_dir: Path, size: str) -> str | None:
    """Per-base mean ± std over seed replicates and per-seed paired diffs; None if no replicates."""
    groups = _seed_groups(runs, size)
    if not groups:
        return None
    out = [SEED_CAPTION, "", _table(["base", "n seeds", *SEED_METRICS], _seed_rows(groups))]
    prows, notes = _seed_paired_rows(groups, results_dir, size)
    if prows:
        header = ["comparison", "seed", "metric", "mean diff (summary: mean ± std)", "95% CI"]
        out += ["", "_Paired by seed: same seed N on both sides, per-user diffs over shared users._",
                "", _table(header, prows)]  # fmt: skip
    out += ["", *(f"- {n}" for n in notes)] if notes else []
    return "\n".join(out)


def _safe_load_runs(results_dir: Path) -> list[dict]:
    """Run records that parse and have the keys the tables need; bad files are logged."""
    runs = []
    for p in sorted(results_dir.glob("*.json")) if results_dir.is_dir() else []:
        try:
            r = json.loads(p.read_text())
            if not (r["run_name"] and r["timestamp"] and isinstance(r["metrics"], dict)):
                raise ValueError("incomplete record")
            if r.get("extra") is None:
                r["extra"] = {}
            runs.append(r)
        except Exception as e:  # noqa: BLE001
            log.warning("skipping unreadable run record %s: %s", p.name, e)
    return sorted(runs, key=lambda r: r["timestamp"])


def _section(fn, *args) -> str:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        log.warning("section %s failed: %s", getattr(fn, "__name__", fn), e)
        return "_not available (could not be rendered; see log)_"


def write_readme_tables(results_dir, out_md) -> Path:
    """Write headline, ablation, slice, fairness and paired-diff sections to `out_md`.

    Every section is rendered defensively: a bad record or file yields a "not available" row.
    """
    results_dir, out_md = Path(results_dir), Path(out_md)
    runs = _safe_load_runs(results_dir)
    sizes = sorted({r["extra"].get("dataset_size") for r in runs} - {None}) or ["50m"]
    evals_all = list(_latest(runs, "test", None, evals=True).values())
    evals_main = [r for r in evals_all if not _is_replicate_run(r["run_name"])]
    parts = ["# Results", ""]
    seeds = {size: _section(_seed_section, runs, results_dir, size) for size in sizes}
    seeds = {k: v for k, v in seeds.items() if v is not None}
    for size in sizes:
        parts += [
            f"## Headline ({size}, test split, Listen+ truth and warm users)",
            "",
            _section(lambda s=size: results_table(runs, size=s)),
            "",
        ]
        if size in seeds:
            parts += ["_Seed replicates (`<run>_sN`) are summarized below, not listed here._", ""]
        parts += [RECALL_FOOTNOTE, ""]
    for size in sizes:
        parts += [
            f"## Ablations ({size})",
            "",
            _section(lambda s=size: ablation_table(runs, size=s)),
            "",
        ]
    parts += ["## Slices", "", SLICE_FOOTNOTE, ""]
    for size in sizes:
        evals = [r for r in evals_all if r["extra"].get("dataset_size") == size]
        if not evals:
            parts += [f"### {size}", "", _section(slice_table, None), ""]
        for r in evals:
            name = r["run_name"].removesuffix("-eval")
            parts += [f"### {name} ({size})", "", _section(slice_table, r), ""]
    parts += ["## Fairness and confidence intervals", ""]
    for size in sizes:
        evals = [r for r in evals_all if r["extra"].get("dataset_size") == size]
        parts += [f"### {size}", "", _section(fairness_table, evals), ""]
    parts += ["## Paired difference: HookSeq vs SASRec", ""]
    for size in sizes:
        parts += [f"### {size}", "", _section(_paired_section, evals_main, results_dir, size), ""]
    if seeds:
        parts += ["## Seed replicates", ""]
        for size, body in seeds.items():
            parts += [f"### {size}", "", body, ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts))
    return out_md
