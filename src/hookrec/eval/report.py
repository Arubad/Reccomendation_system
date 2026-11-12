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


def _latest(
    runs: list[dict], split: str, size: str | None, evals: bool = False
) -> dict[tuple[str, str | None], dict]:
    """Latest record per (run_name, size) for `split` (and dataset size, if given)."""
    out: dict[str, dict] = {}
    for r in sorted(runs, key=lambda r: r["timestamp"]):
        ex = r.get("extra", {})
        if _is_eval(r) != evals or ex.get("split") != split:
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
    return {m: v for m, v in (ref.get(str(size).lower()) or {}).items()}


def results_table(
    runs: list[dict],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    split: str = "test",
    size: str = "50m",
) -> str:
    """Latest run per name for `split` and `size`, then the paper rows marked "(paper)"."""
    rows = []
    for (name, _), r in sorted(
        _latest(runs, split, size).items(), key=lambda kv: (kv[0][0], str(kv[0][1]))
    ):
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


SLICE_KEYS = ("ndcg@10", "recall@10", "ndcg@100", "recall@100", "coverage@10")


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
        rows.append([name, int(m.get("n_users", 0)), *(_fmt(m.get(c)) for c in SLICE_KEYS)])
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


def _paired_section(evals: list[dict], results_dir: Path, size: str) -> str:
    """HookSeq vs SASRec paired diff for one size; needs matching size and processed_dir."""
    header = ["metric", "mean diff", "95% CI"]
    by = {r["run_name"]: r for r in evals if r.get("extra", {}).get("dataset_size") == size}
    a_rec = by.get("hookseq-test-eval")
    b_rec = by.get("a0_sasrec-test-eval") or by.get("sasrec-test-eval")
    if a_rec is None or b_rec is None:
        return _table(header, [[f"HookSeq vs SASRec: {NOT_RUN}", "-", "-"]])
    if a_rec["extra"].get("processed_dir") != b_rec["extra"].get("processed_dir"):
        return _table(
            header, [["HookSeq vs SASRec: not available (processed_dir mismatch)", "-", "-"]]
        )
    a = results_dir / f"hookseq-test-{size}-eval-peruser.npz"
    b = results_dir / f"{b_rec['run_name'].removesuffix('-eval')}-{size}-eval-peruser.npz"
    rows = []
    for m in ("ndcg@10", "recall@100"):
        try:
            mean, lo, hi = paired_diff_ci(a, b, m)
        except Exception as e:  # noqa: BLE001  # missing / corrupt npz
            log.warning("paired diff %s unavailable: %s", m, e)
            rows.append([f"{m}: not available", "-", "-"])
            continue
        label = b_rec["run_name"].removesuffix("-test-eval")
        rows.append([f"{m} (HookSeq - {label})", f"{mean:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
    return _table(header, rows)


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
    parts = ["# Results", ""]
    for size in sizes:
        parts += [
            f"## Headline ({size}, test split, Listen+)",
            "",
            _section(lambda s=size: results_table(runs, size=s)),
            "",
        ]
    for size in sizes:
        parts += [
            f"## Ablations ({size})",
            "",
            _section(lambda s=size: ablation_table(runs, size=s)),
            "",
        ]
    parts += ["## Slices", ""]
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
        parts += [f"### {size}", "", _section(_paired_section, evals_all, results_dir, size), ""]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(parts))
    return out_md
