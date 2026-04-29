"""Step 6: per-SNP and per-block explanations.

Pulls together SHAP values from the group ranker (TreeExplainer on XGBoost),
gradient×input scores from the SNP-level MLP (already produced inside
`SnpRanker`), and overlapping functional annotations into a single tidy
output table that the proposal's Step 6 calls for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)


def shap_for_group_model(
    model, block_features: pd.DataFrame, top_blocks: list[str]
) -> pd.DataFrame:
    """SHAP values per block for the top-ranked blocks.

    Returns long-format `(block_id, feature, shap)`.
    """
    explainer = shap.TreeExplainer(model)
    sub = block_features.loc[block_features.index.isin(top_blocks)]
    if sub.empty:
        return pd.DataFrame(columns=["block_id", "feature", "shap"])
    sv = explainer.shap_values(sub.to_numpy())
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    rows = []
    for i, block_id in enumerate(sub.index):
        for j, feat in enumerate(sub.columns):
            rows.append((block_id, feat, float(sv[i, j])))
    return pd.DataFrame(rows, columns=["block_id", "feature", "shap"])


def assemble_selected_table(
    selected_snps: pd.DataFrame,
    snp_assignments: pd.DataFrame,
    pi_blocks: pd.Series,
    pi_snps: pd.Series,
    snp_scores: pd.Series,
    snp_features: pd.DataFrame,
    gwas_catalog_hits: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Persist the proposal's Step-6 output: ranked SNPs with full provenance."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = selected_snps[["rsid"]].copy()

    # Block context.
    base = base.merge(snp_assignments[["rsid", "chrom", "pos", "block_id"]], on="rsid", how="left")

    # Stability scores.
    base["pi_snp"] = base["rsid"].map(pi_snps).fillna(0.0)
    base["pi_block"] = base["block_id"].map(pi_blocks).fillna(0.0)

    # Model score.
    base["snp_score"] = base["rsid"].map(snp_scores).fillna(0.0)

    # Annotation summary as JSON (preserves arbitrary annotation columns).
    feat_lookup = snp_features.to_dict(orient="index")
    base["annotations_json"] = base["rsid"].map(
        lambda r: json.dumps(feat_lookup.get(r, {}), default=float)
    )

    # GWAS Catalog match.
    catalog_set = set(gwas_catalog_hits["rsid"].astype(str))
    base["gwas_catalog_match"] = base["rsid"].astype(str).isin(catalog_set)
    catalog_traits = (
        gwas_catalog_hits.groupby("rsid")["trait"].agg(lambda s: ";".join(sorted(set(s))))
    )
    base["gwas_catalog_traits"] = base["rsid"].map(catalog_traits).fillna("")

    base = base.sort_values(["pi_snp", "snp_score"], ascending=False)
    base.to_parquet(out_path, index=False)
    log.info("Wrote selected SNP table (%d rows) to %s", len(base), out_path)
    return out_path
