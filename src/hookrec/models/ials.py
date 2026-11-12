"""iALS baseline on top of `implicit`, with a MostPop fallback for users unseen at fit time."""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csr_matrix

from hookrec.data.label import LIKE, LISTEN_FULL
from hookrec.data.processed import ProcessedData
from hookrec.data.sequences import PackedSequences
from hookrec.models.base import Split, history_for, register
from hookrec.models.popularity import MostPop, fill_cold

log = logging.getLogger(__name__)


class IALS:
    """Implicit-feedback ALS over positive counts.

    Confidence is applied exactly once, here: the matrix handed to `implicit` holds
    `1 + alpha * count` on non-zeros, and `implicit`'s own `alpha` is left at 1.0
    (in implicit 0.7 a non-1 `alpha` would multiply the matrix again inside `fit`).
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        alpha: float = 10.0,
        iterations: int = 15,
        use_gpu: bool | None = None,
        seed: int = 0,
    ):
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.use_gpu = use_gpu
        self.seed = int(seed)

    def _confidence_matrix(self, seqs: PackedSequences, n_items: int) -> csr_matrix:
        pos = (seqs.tokens == LISTEN_FULL) | (seqs.tokens == LIKE)
        rows = np.repeat(np.arange(len(seqs.uids), dtype=np.int64), np.diff(seqs.offsets))[pos]
        cols = np.asarray(seqs.items)[pos].astype(np.int64)
        mat = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(seqs.uids), n_items + 1),
        )  # duplicate (row, col) entries are summed, giving positive counts
        mat.data = 1.0 + self.alpha * mat.data
        return mat

    def fit(self, seqs: PackedSequences, n_items: int) -> IALS:
        import implicit.gpu
        from implicit.als import AlternatingLeastSquares

        use_gpu = implicit.gpu.HAS_CUDA if self.use_gpu is None else bool(self.use_gpu)
        log.info("iALS backend: %s", "gpu" if use_gpu else "cpu")
        self._seqs = seqs
        self._user_items = self._confidence_matrix(seqs, n_items)
        kw = {} if use_gpu else {"num_threads": 0}  # 0 = all cores
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=1.0,
            iterations=self.iterations,
            use_gpu=use_gpu,
            random_state=self.seed,
            **kw,
        )
        self._model.fit(self._user_items, show_progress=False)
        self._fallback = MostPop().fit(seqs, n_items)
        return self

    def recommend(self, uids: np.ndarray, k: int) -> np.ndarray:
        uids = np.asarray(uids)
        rows = self._seqs.index_of(uids)
        # users without a positive (empty CSR row) get the fallback too: their factors are arbitrary
        has_pos = np.diff(self._user_items.indptr) > 0
        known = (rows >= 0) & has_pos[np.maximum(rows, 0)]
        out = np.zeros((len(uids), k), dtype=np.int64)
        if known.any():
            r = rows[known]
            ids, _ = self._model.recommend(
                r, self._user_items[r], N=k + 1, filter_already_liked_items=False
            )
            ids = np.asarray(ids.to_numpy() if hasattr(ids, "to_numpy") else ids, dtype=np.int64)
            # drop the pad item 0 wherever it ranks, keep the first k, 0-pad the remainder
            keep = ids != 0
            order = np.argsort(~keep, axis=1, kind="stable")
            ids = np.where(
                np.take_along_axis(keep, order, axis=1), np.take_along_axis(ids, order, axis=1), 0
            )[:, :k]
            out[np.flatnonzero(known), : ids.shape[1]] = ids
        return fill_cold(uids, out, self._fallback.top_items(k), known)


@register("ials")
def _build_ials(data: ProcessedData, cfg: dict, split: Split) -> IALS:
    seqs, _ = history_for(data, split)
    model = IALS(
        factors=cfg["factors"],
        regularization=cfg["regularization"],
        alpha=cfg["alpha"],
        iterations=cfg["iterations"],
        use_gpu=cfg.get("use_gpu"),
        seed=cfg.get("seed", 0),
    )
    return model.fit(seqs, data.vocab.n_items)
