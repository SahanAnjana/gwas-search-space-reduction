"""Step 3a: rank LD blocks by gradient-boosted classification.

The target for each block on a given training fold is the within-fold
leave-block-out signal: did at least one SNP inside the block reach a single-
SNP χ² p < threshold? This converts block ranking into a supervised binary
problem with a clear, leakage-controlled label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from clear_a._chi2_vec import vectorized_chi2_pvals

log = logging.getLogger(__name__)


@dataclass
class GroupRankConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    block_label_pvalue: float = 1e-4
    seed: int = 42


def _block_labels_from_chisq(
    dosages: np.ndarray,
    y: np.ndarray,
    snp_assignments: pd.DataFrame,
    threshold_p: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute single-SNP χ² for each SNP in the training fold and label blocks.

    Uses 2x2 contingency on dosage > 0 (carrier) vs phenotype label. This is
    a lightweight proxy for the standard additive test — sufficient for
    *labeling* blocks, not for inference.
    """
    n_samples, n_snps = dosages.shape
    pvals = vectorized_chi2_pvals(dosages, y)
    snp_pvals = pd.DataFrame({
        "rsid": snp_assignments["rsid"].values[:n_snps],
        "pvalue": pvals,
    })
    merged = snp_assignments.merge(snp_pvals, on="rsid", how="inner")
    block_min_p = merged.groupby("block_id")["pvalue"].min()
    block_labels = (block_min_p < threshold_p).astype(int)
    return block_labels, snp_pvals


class GroupRanker:
    def __init__(self, config: GroupRankConfig | None = None):
        self.config = config or GroupRankConfig()
        self.model: xgb.XGBClassifier | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        block_features: pd.DataFrame,
        dosages: np.ndarray,
        y: np.ndarray,
        snp_assignments: pd.DataFrame,
    ) -> "GroupRanker":
        labels, _ = _block_labels_from_chisq(
            dosages, y, snp_assignments, self.config.block_label_pvalue
        )
        # Align block_features rows to label index; missing labels (blocks with
        # no SNPs in the fold) are dropped.
        aligned = block_features.join(labels.rename("y"), how="inner")
        X = aligned.drop(columns=["y"])
        y_block = aligned["y"].astype(int).to_numpy()
        if y_block.sum() == 0 or y_block.sum() == len(y_block):
            log.warning("Group-level labels are degenerate (all %d).", y_block[0])
        self.feature_names_ = list(X.columns)
        pos_rate = max(y_block.mean(), 1e-3)
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(1 - pos_rate) / pos_rate,
            random_state=self.config.seed,
            n_jobs=0,
            tree_method="hist",
        )
        self.model.fit(X.to_numpy(), y_block)
        log.info("Trained GroupRanker on %d blocks (pos=%d)", len(y_block), int(y_block.sum()))
        return self

    def rank(self, block_features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("GroupRanker not fitted")
        X = block_features.reindex(columns=self.feature_names_, fill_value=0.0)
        scores = self.model.predict_proba(X.to_numpy())[:, 1]
        return pd.Series(scores, index=block_features.index, name="block_score").sort_values(
            ascending=False
        )

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("GroupRanker not fitted")
        imps = self.model.feature_importances_
        return pd.Series(imps, index=self.feature_names_, name="importance").sort_values(
            ascending=False
        )
