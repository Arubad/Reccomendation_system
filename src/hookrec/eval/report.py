"""Markdown results tables from run records (`save_run` JSONs) and the paper reference numbers."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import yaml

from hookrec.utils.results import load_runs

PAPER_REFERENCE = Path(__file__).resolve().parents[3] / "configs" / "paper_reference.yaml"
DEFAULT_METRICS = ("ndcg@10", "ndcg@100", "recall@10", "recall@100", "coverage@10")
NOT_RUN = "not yet run"


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) and not math.isnan(x) else "-"


def _row(cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join([_row(header), "|" + "---|" * len(header), *(_row(r) for r in rows)])


def _is_eval(r: dict) -> bool:
    return r["run_name"].endswith("-eval") or r.get("extra", {}).get("kind") == "eval"


def _latest(runs: list[dict], split: str, size: str | None, evals: bool = False) -> dict[str, dict]:
    """Latest record per run_name for `split` (and dataset size, if given)."""
    out: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: r["timestamp"]):
        ex = r.get("extra", {})
        if _is_eval(r) != evals or ex.get("split") != split:
            continue
        if size is not None and ex.get("dataset_size") != size:
            continue
        out[r["run_name"]] = r
    return out


def paper_rows(size: str, path: Path = PAPER_REFERENCE) -> dict[str, dict]:
    """Paper reference numbers for `size` (model -> metric -> value); empty if unavailable."""
    if not Path(path).exists():
        return {}
    ref = yaml.safe_load(Path(path).read_text()) or {}
    return {m: v for m, v in (ref.get(str(size).lower()) or {}).items()}


def results_table(
    runs: list[dict],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    split: str = "test",
    size: str = "50m",
) -> str:
    """Latest run per name for `split` and `size`, then the paper rows marked "(paper)"."""
    rows = []
    for name, r in sorted(_latest(runs, split, size).items()):
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
    for name, r in _latest(runs, split, size).items():
        m = pat.match(name)
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


def slice_table(eval_run: dict | None) -> str:
    """Per-slice metrics of one eval record (`extra.slices`)."""
    header = ["slice", "users", "ndcg@10", "recall@10", "ndcg@100", "recall@100", "coverage@10"]
    sl = (eval_run or {}).get("extra", {}).get("slices")
    if not sl:
        return _table(header, [[NOT_RUN, "-", "-", "-", "-", "-", "-"]])
    rows = []
    for name, m in sl.items():
        rows.append([name, int(m.get("n_users", 0)), *(_fmt(m.get(c)) for c in header[2:])])
    cold = eval_run["extra"].get("metrics_cold") or {}
    if cold:
        rows.append(
            [f"cold users (n={eval_run['extra'].get('n_cold_users', '?')})", "-"]
            + [_fmt(cold.get(c)) for c in header[2:]]
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


def _paired_section(results_dir: Path) -> str:
    a = results_dir / "hookseq-test-eval-peruser.npz"
    b = next(
        (
            p
            for p in (
                results_dir / "a0_sasrec-test-eval-peruser.npz",
                results_dir / "sasrec-test-eval-peruser.npz",
            )
            if p.exists()
        ),
        None,
    )
    header = ["metric", "mean diff", "95% CI"]
    if not a.exists() or b is None:
        return _table(header, [[f"HookSeq vs SASRec: {NOT_RUN}", "-", "-"]])
    rows = []
    for m in ("ndcg@10", "recall@100"):
        mean, lo, hi = paired_diff_ci(a, b, m)
        rows.append([f"{m} (HookSeq - {b.name.split('-test')[0]})", f"{mean:+.4f}",
                     f"[{lo:+.4f}, {hi:+.4f}]"])  # fmt: skip
    return _table(header, rows)


def write_readme_tables(results_dir, out_md) -> Path:
    """Write headline, ablation, slice, fairness and paired-diff sections to `out_md`."""
    results_dir, out_md = Path(results_dir), Path(out_md)
    runs = load_runs(results_dir)
    sizes = sorted({r["extra"].get("dataset_size") for r in runs if r.get("extra")} - {None})
    sizes = sizes or ["50m"]
    parts = ["# Results", ""]
    for size in sizes:
        parts += [
            f"## Headline ({size}, test split, Listen+)",
            "",
            results_table(runs, size=size),
            "",
        ]
    for size in sizes:
        parts += [f"## Ablations ({size})", "", ablation_table(runs, size=size), ""]
    evals = list(_latest(runs, "test", None, evals=True).values())
    parts += ["## Slices", ""]
    if not evals:
        parts += [slice_table(None), ""]
    for r in evals:
        parts += [f"### {r['run_name'].removesuffix('-eval')}", "", slice_table(r), ""]
    parts += ["## Fairness and confidence intervals", "", fairness_table(evals), ""]
    parts += ["## Paired difference: HookSeq vs SASRec", "", _paired_section(results_dir), ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts))
    return out_md
