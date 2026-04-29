"""Evaluation metrics: GWAS Catalog recovery, redundancy, transferability.

These are the metrics the proposal commits to in its problem statement,
adapted to a real-data-only setting where ground truth is the GWAS Catalog
plus the project's own LD-aware redundancy and cross-ancestry checks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_gwas_catalog_hits(catalog_tsv: Path, traits: list[str]) -> pd.DataFrame:
    """Filter the GWAS Catalog associations download to the supplied trait list.

    The official download is `gwas_catalog_v1.0.2-associations_*.tsv`. We
    match `MAPPED_TRAIT` and `DISEASE/TRAIT` case-insensitively against any
    substring of the requested traits — broad on purpose, since trait
    spellings drift.
    """
    df = pd.read_csv(catalog_tsv, sep="\t", low_memory=False)
    rename = {
        "STRONGEST SNP-RISK ALLELE": "snp_risk_allele",
        "SNPS": "rsid",
        "MAPPED_TRAIT": "mapped_trait",
        "DISEASE/TRAIT": "trait",
    }
    df = df.rename(columns=rename)
    if "mapped_trait" not in df.columns:
        # The non-ontology-annotated catalog drops MAPPED_TRAIT; substitute
        # `trait` so the substring search still has something to match against.
        df["mapped_trait"] = df["trait"]
    keep = pd.Series(False, index=df.index)
    for t in traits:
        t_low = t.lower()
        keep |= df["trait"].fillna("").str.lower().str.contains(t_low, regex=False)
        keep |= df["mapped_trait"].fillna("").str.lower().str.contains(t_low, regex=False)
    df = df.loc[keep, ["rsid", "trait", "mapped_trait", "P-VALUE"]].copy()
    df["rsid"] = df["rsid"].astype(str).str.split(";").str[0].str.strip()
    df = df[df["rsid"].str.startswith("rs")]
    df = df.drop_duplicates("rsid").reset_index(drop=True)
    log.info("Loaded %d GWAS Catalog hits for traits=%s", len(df), traits)
    return df


def recovery_curve(
    ranked_rsids: pd.Series,
    catalog_hits: pd.DataFrame,
    ks: list[int],
) -> pd.DataFrame:
    """For each K, report how many catalog hits appear in the top-K of `ranked_rsids`."""
    catalog = set(catalog_hits["rsid"].astype(str))
    rsids = ranked_rsids.astype(str).tolist()
    rows = []
    for k in ks:
        topk = set(rsids[:k])
        hit = topk & catalog
        rows.append({
            "k": k,
            "n_catalog_hits_recovered": len(hit),
            "recall": len(hit) / max(len(catalog), 1),
            "precision": len(hit) / max(k, 1),
        })
    return pd.DataFrame(rows)


def redundancy_r2(selected_rsids: list[str], ld_matrix: pd.DataFrame) -> float:
    """Mean pairwise r² among selected SNPs, using a precomputed LD matrix.

    `ld_matrix` should be square, indexed and columned by rsid. Missing pairs
    are treated as zero (no LD info available).
    """
    sel = [r for r in selected_rsids if r in ld_matrix.index]
    if len(sel) < 2:
        return 0.0
    sub = ld_matrix.loc[sel, sel].to_numpy()
    iu = np.triu_indices(len(sel), k=1)
    return float(np.nanmean(sub[iu]))


def transferability_lookup(
    selected_rsids: list[str],
    non_eur_summary: pd.DataFrame,
    p_thresholds: tuple[float, ...] = (5e-8, 1e-5, 0.05),
) -> pd.DataFrame:
    """For each threshold, fraction of selected SNPs that replicate in non-EUR summary stats.

    `non_eur_summary` must have columns `rsid, pvalue, ancestry`.
    """
    sel = set(selected_rsids)
    rows = []
    for ancestry, sub in non_eur_summary.groupby("ancestry"):
        sub_sel = sub[sub["rsid"].isin(sel)]
        n_avail = len(sub_sel)
        for thr in p_thresholds:
            n_rep = int((sub_sel["pvalue"] < thr).sum())
            rows.append({
                "ancestry": ancestry,
                "p_threshold": thr,
                "n_available": n_avail,
                "n_replicated": n_rep,
                "fraction_replicated": n_rep / n_avail if n_avail else 0.0,
            })
    return pd.DataFrame(rows)
