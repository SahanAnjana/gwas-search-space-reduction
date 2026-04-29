"""Vectorized Pearson chi-square 2x2 contingency tests over many SNPs.

Replaces the per-SNP `scipy.stats.fisher_exact` loop in `inference.py` and
`models/group_ranker.py`. For 1.2M SNPs and ~4k samples the loop took ~50 s
per fit; the vectorized version runs in ~1-2 s.

The closed-form 2x2 chi-square (no continuity correction) is

    chi2 = N * (a*e - b*d)**2 / [(a+b)*(d+e)*(a+d)*(b+e)]

where `(a,b,d,e)` are the cell counts of (carrier, case), (carrier, control),
(non-carrier, case), (non-carrier, control). Constant SNPs (any zero margin)
get chi2 = 0 → p = 1, matching the loop's previous behaviour.

For sparse tables Pearson chi-square diverges from Fisher's exact, but in
permutation-calibrated workflows (Westfall-Young) the same screening
statistic is used for observed and null draws so the discrepancy cancels.
For the block-labelling use case in `group_ranker` we only need a screening
threshold, so the speedup more than justifies the approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as _chi2


def vectorized_chi2_pvals(
    dosages: np.ndarray,
    y: np.ndarray,
    chunk: int = 50_000,
    carrier_threshold: float = 0.5,
) -> np.ndarray:
    """Compute 2x2 chi-square p-values for every SNP in `dosages`.

    Parameters
    ----------
    dosages : (n_samples, n_snps) array of genotype dosages (0..2). int8 or float.
    y       : (n_samples,) binary phenotype (0/1). Other ints are coerced to >0.
    chunk   : number of SNPs to process per pass (memory control).
    carrier_threshold : dosage > this counts as a carrier.

    Returns
    -------
    pvals : (n_snps,) float32 array. SNPs with any zero margin get p=1.
    """
    n_samples, n_snps = dosages.shape
    y_bin = (np.asarray(y) > 0).astype(np.int32)
    N = int(n_samples)
    col1 = int(y_bin.sum())  # cases
    col2 = N - col1

    if col1 == 0 or col2 == 0:
        return np.ones(n_snps, dtype=np.float32)

    pvals = np.empty(n_snps, dtype=np.float32)
    for start in range(0, n_snps, chunk):
        stop = min(start + chunk, n_snps)
        block = np.asarray(dosages[:, start:stop])
        carriers = (block > carrier_threshold).astype(np.int32)
        # a = #(carrier=1 & case=1) per SNP — matrix-vector product.
        a = carriers.T @ y_bin
        carrier_count = carriers.sum(axis=0).astype(np.int64)
        b = carrier_count - a
        d = col1 - a
        e = col2 - b
        row1 = carrier_count
        row2 = N - row1
        denom = row1.astype(np.float64) * row2 * col1 * col2
        num = (a.astype(np.float64) * e - b.astype(np.float64) * d) ** 2
        chi2_stat = np.where(denom > 0, N * num / np.where(denom > 0, denom, 1), 0.0)
        block_pvals = _chi2.sf(chi2_stat, df=1).astype(np.float32)
        # Constant SNPs (denom=0) get p=1 explicitly.
        block_pvals[denom == 0] = 1.0
        pvals[start:stop] = block_pvals
    return pvals
