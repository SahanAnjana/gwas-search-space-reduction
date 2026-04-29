"""Bootstrap stability selection for blocks and SNPs.

Wraps a single-fold "fit + rank" callable in a bootstrap loop and records
how often each entity (block or SNP) appears in the top-K. The selection
frequency π is the empirical probability of appearing in the top-K under
resampling — a stability proxy that avoids the overconfidence of any one
random split.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class StabilityConfig:
    n_bootstraps: int = 100
    sample_fraction: float = 0.8
    top_k: int = 1000
    seed: int = 42


def bootstrap_selection(
    fit_rank_fn: Callable[[np.ndarray], pd.Series],
    n_samples: int,
    config: StabilityConfig | None = None,
    top_k_override: int | None = None,
) -> pd.Series:
    """Run bootstrap selection.

    `fit_rank_fn(sample_idx)` must accept a row-index array, refit the
    pipeline on those rows, and return a Series of scores indexed by entity
    id (block_id or rsid). The top-K from each bootstrap are tallied; the
    returned Series is selection frequency π in [0, 1].

    `top_k_override` lets callers pick a top-K appropriate to the entity
    count (e.g. for block-level bootstraps where the universe is small).
    """
    cfg = config or StabilityConfig()
    rng = np.random.default_rng(cfg.seed)
    sample_size = int(cfg.sample_fraction * n_samples)
    counts: Counter[str] = Counter()
    for b in range(cfg.n_bootstraps):
        idx = rng.choice(n_samples, size=sample_size, replace=False)
        scores = fit_rank_fn(idx)
        # If top_k exceeds the entity universe, the metric collapses to 1.0
        # for every entity — informative only when capped tighter than N.
        k = top_k_override if top_k_override is not None else cfg.top_k
        k = min(k, max(1, len(scores) // 3))  # keep π informative
        top = scores.nlargest(k).index
        counts.update(top)
        log.info("Bootstrap %d/%d done (top entity: %s, score=%.4g)",
                 b + 1, cfg.n_bootstraps,
                 scores.index[0] if len(scores) else "(empty)",
                 float(scores.iloc[0]) if len(scores) else 0.0)
    pi = pd.Series(
        {k: v / cfg.n_bootstraps for k, v in counts.items()},
        name="pi",
    ).sort_values(ascending=False)
    return pi


def threshold_stable(pi_blocks: pd.Series, pi_snps: pd.Series, pi_block_min: float, pi_snp_min: float) -> pd.DataFrame:
    """Combine block-level and SNP-level stability into a final selection set."""
    stable_blocks = set(pi_blocks[pi_blocks >= pi_block_min].index)
    stable_snps = pi_snps[pi_snps >= pi_snp_min]
    out = stable_snps.to_frame("pi_snp").reset_index().rename(columns={"index": "rsid"})
    out["block_stable"] = out["rsid"].isin(stable_blocks)
    log.info(
        "Stable selection: %d SNPs (π≥%.2f), %d blocks (π≥%.2f)",
        len(out), pi_snp_min, len(stable_blocks), pi_block_min,
    )
    return out
