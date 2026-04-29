from __future__ import annotations

import numpy as np

from clear_a.inference import westfall_young_permutation


def test_westfall_young_threshold_holds_under_null() -> None:
    rng = np.random.default_rng(0)
    n_samples = 200
    n_snps = 30
    dosages = (rng.random((n_samples, n_snps)) < 0.3).astype(np.float32) * 2
    y = rng.integers(0, 2, size=n_samples)
    result = westfall_young_permutation(
        dosages, y, [f"rs{i}" for i in range(n_snps)], n_perms=200, fdr=0.05, rng=rng,
    )
    # Under the null, family-wise FDR should hold the nominal level (with slack).
    assert result.table["selected"].sum() <= int(0.10 * n_snps + 1)
    assert (result.table["pvalue"] >= 0).all()
    assert (result.table["pvalue"] <= 1).all()


def test_westfall_young_detects_planted_signal() -> None:
    rng = np.random.default_rng(1)
    n_samples = 400
    n_snps = 50
    dosages = (rng.random((n_samples, n_snps)) < 0.3).astype(np.float32) * 2
    y = rng.integers(0, 2, size=n_samples)
    # Plant a strong association on SNP 0.
    dosages[:, 0] = (y * 2).astype(np.float32) + rng.normal(0, 0.1, n_samples).astype(np.float32)
    dosages[:, 0] = np.clip(dosages[:, 0], 0, 2)
    result = westfall_young_permutation(
        dosages, y, [f"rs{i}" for i in range(n_snps)], n_perms=200, fdr=0.05, rng=rng,
    )
    assert result.table.loc[0, "selected"], "Planted signal should be flagged"
