"""Post-selection FDR control.

Two procedures, both with the same return surface (a DataFrame with one row
per SNP and a `selected` boolean column):

- Per-LD-block second-order Gaussian knockoffs via `knockpy`. Avoids global
  covariance estimation by working block-wise, which both fits the project's
  LD-block structure and dodges the conditioning-number problems that hit
  whole-genome knockoffs.
- Westfall–Young permutation (the COMBI-style approach). Slower but assumption-
  light; useful as a comparator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from clear_a._chi2_vec import vectorized_chi2_pvals

log = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    table: pd.DataFrame  # rsid, score, threshold, selected
    method: str
    fdr: float


def westfall_young_permutation(
    dosages: np.ndarray,
    y: np.ndarray,
    rsids: list[str],
    n_perms: int = 1000,
    fdr: float = 0.05,
    rng: np.random.Generator | None = None,
) -> InferenceResult:
    """Westfall–Young permutation calibration of single-SNP χ² p-values.

    Computes observed p-values, then for each permutation of `y` records the
    minimum p-value over all SNPs. The (1 - fdr) quantile of those minima
    becomes the family-wise threshold.
    """
    rng = rng or np.random.default_rng(42)
    n_samples, n_snps = dosages.shape
    if n_snps == 0:
        log.warning("Westfall-Young called with 0 SNPs; returning empty result.")
        return InferenceResult(
            table=pd.DataFrame({"rsid": [], "score": [], "pvalue": [], "threshold_p": [], "selected": []}),
            method="westfall_young", fdr=fdr,
        )
    y_bin = (y > 0).astype(np.int32)

    obs_p = vectorized_chi2_pvals(dosages, y_bin)
    perm_min: list[float] = []
    for k in range(n_perms):
        y_perm = rng.permutation(y_bin)
        perm_min.append(float(vectorized_chi2_pvals(dosages, y_perm).min()))
        if (k + 1) % max(1, n_perms // 10) == 0:
            log.info("WY permutation %d/%d", k + 1, n_perms)
    threshold = float(np.quantile(perm_min, fdr))
    table = pd.DataFrame({
        "rsid": rsids,
        "score": -np.log10(np.clip(obs_p.astype(np.float64), 1e-300, 1.0)),
        "pvalue": obs_p,
        "threshold_p": threshold,
        "selected": obs_p < threshold,
    })
    log.info(
        "Westfall-Young: threshold p=%.3g, %d selected",
        threshold,
        int(table["selected"].sum()),
    )
    return InferenceResult(table=table, method="westfall_young", fdr=fdr)


def knockoff_filter_per_block(
    dosages: np.ndarray,
    y: np.ndarray,
    rsids: list[str],
    snp_assignments: pd.DataFrame,
    fdr: float = 0.05,
    seed: int = 42,
) -> InferenceResult:
    """Run per-LD-block model-X Gaussian knockoffs and apply the knockoff filter.

    Block-wise knockoffs use only the within-block covariance, which is well-
    conditioned even on consumer-array data. For each block we construct
    knockoffs, run a Lasso feature-statistic, then aggregate W-statistics
    across blocks for one knockoff filter call.
    """
    try:
        from knockpy.knockoffs import GaussianSampler
        from knockpy.knockoff_stats import LassoStatistic
        from knockpy.knockoff_filter import KnockoffFilter
    except ImportError as e:
        raise ImportError("knockpy is required for the knockoff filter") from e

    rng = np.random.default_rng(seed)
    rsid_idx = {r: i for i, r in enumerate(rsids)}
    W_full = np.zeros(len(rsids), dtype=np.float64)

    for block_id, group in snp_assignments.groupby("block_id"):
        cols = [rsid_idx[r] for r in group["rsid"] if r in rsid_idx]
        if len(cols) < 5:  # too small for stable knockoff construction
            continue
        Xb = dosages[:, cols].astype(np.float64)
        Xb = Xb - Xb.mean(axis=0, keepdims=True)
        try:
            sampler = GaussianSampler(X=Xb, method="sdp")
            Xk = sampler.sample_knockoffs()
            stat = LassoStatistic()
            W = stat.fit(X=Xb, Xk=Xk, y=y.astype(np.float64))
        except Exception as e:  # noqa: BLE001 — knockoff construction can fail on near-singular blocks
            log.warning("Knockoff construction failed for block %s: %s", block_id, e)
            continue
        for col, w in zip(cols, W):
            W_full[col] = w

    # Apply the knockoff filter threshold to the aggregated W stats.
    abs_w = np.abs(W_full)
    sorted_w = np.sort(abs_w[abs_w > 0])[::-1]
    threshold = np.inf
    for t in sorted_w:
        denom = max((W_full >= t).sum(), 1)
        ratio = (1 + (W_full <= -t).sum()) / denom
        if ratio <= fdr:
            threshold = t
            break
    selected = W_full >= threshold
    table = pd.DataFrame({
        "rsid": rsids,
        "score": W_full,
        "threshold_w": threshold,
        "selected": selected,
    })
    log.info(
        "Knockoff filter: threshold W=%.4g, %d selected",
        float(threshold) if np.isfinite(threshold) else float("inf"),
        int(table["selected"].sum()),
    )
    return InferenceResult(table=table, method="knockoff_per_block", fdr=fdr)


def empirical_fdr_under_null(
    inference_fn: Callable[[np.ndarray, np.ndarray], InferenceResult],
    dosages: np.ndarray,
    y: np.ndarray,
    n_replicates: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate empirical FDR under the null by permuting phenotype labels.

    Useful as the "synthetic-free null check" listed in the plan's verification
    section: under permuted phenotype, every selection is by definition false,
    so the realized selection rate is the empirical FDR.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for r in range(n_replicates):
        y_perm = rng.permutation(y)
        result = inference_fn(dosages, y_perm)
        rows.append({
            "replicate": r,
            "method": result.method,
            "n_selected": int(result.table["selected"].sum()),
            "fraction_selected": float(result.table["selected"].mean()),
            "nominal_fdr": result.fdr,
        })
    return pd.DataFrame(rows)
