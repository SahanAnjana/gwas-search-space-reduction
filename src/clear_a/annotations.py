"""Join external functional annotations onto SNPs and aggregate per LD block.

Sources:
- LDSC baselineLD v2.2: per-SNP indicators across 97 functional categories.
  Distributed as gzipped per-chromosome `.annot.gz` files keyed by rsid.
- GTEx v8 eQTL summary: significant variant-tissue pairs from
  `GTEx_Analysis_v8_eQTL` (variant_id, gene_id, tissue, qval).
- ENCODE / Roadmap chromatin states: BED tracks per cell type, intersected
  by genomic position.

Per-SNP features are concatenated into a numeric matrix; per-block features
are aggregated as means and proportions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_ldsc_baseline(ldsc_dir: Path) -> pd.DataFrame:
    """Concatenate LDSC baselineLD `.annot.gz` files across chromosomes.

    Returns a DataFrame indexed by `rsid` (column SNP) with one numeric column
    per annotation. Drops the redundant CHR/BP/CM columns, since we already
    carry chrom/pos via the SNP info table.
    """
    frames: list[pd.DataFrame] = []
    for chrom in range(1, 23):
        path = ldsc_dir / f"baselineLD.{chrom}.annot.gz"
        if not path.exists():
            log.warning("Missing LDSC baselineLD for chr%d at %s", chrom, path)
            continue
        df = pd.read_csv(path, sep="\t", compression="gzip")
        drop_cols = [c for c in ["CHR", "BP", "CM"] if c in df.columns]
        df = df.drop(columns=drop_cols)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No LDSC baselineLD files found under {ldsc_dir}")
    out = pd.concat(frames, ignore_index=True).set_index("SNP")
    log.info("Loaded LDSC baseline: %d SNPs x %d annotations", *out.shape)
    return out


def load_gtex_eqtl_counts(
    gtex_signif_path: Path, target_tissues: list[str] | None = None
) -> pd.DataFrame:
    """Aggregate GTEx significant eQTL counts per variant.

    `gtex_signif_path` should point to `GTEx_Analysis_v8_eQTL/*.signif_variant_gene_pairs.txt.gz`
    if reading a single tissue, or to a concatenation across tissues with a
    `tissue` column.
    """
    df = pd.read_csv(gtex_signif_path, sep="\t", compression="infer")
    if "tissue" not in df.columns:
        df["tissue"] = gtex_signif_path.stem.split(".")[0]
    if target_tissues:
        df = df[df["tissue"].isin(target_tissues)]
    out = (
        df.groupby("variant_id")
        .agg(
            n_eqtls=("gene_id", "nunique"),
            n_tissues=("tissue", "nunique"),
            min_qval=("qval", "min"),
        )
        .reset_index()
    )
    log.info("GTEx eQTL annotations: %d variants", len(out))
    return out


def build_snp_feature_matrix(
    snp_info: pd.DataFrame,
    ldsc_df: pd.DataFrame | None = None,
    gtex_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble a per-SNP numeric feature matrix indexed by rsid.

    Missing annotations are zero-filled (LDSC indicators, GTEx counts) which
    is the conventional treatment for "no overlap" in genome annotation.
    """
    feats = snp_info.set_index("rsid")[["pos"]].copy()
    feats["maf"] = snp_info.set_index("rsid").get("maf", np.nan)
    feats["log_pos"] = np.log1p(feats["pos"].astype(float))

    if ldsc_df is not None:
        feats = feats.join(ldsc_df, how="left")

    if gtex_df is not None:
        feats = feats.join(
            gtex_df.set_index("variant_id")[["n_eqtls", "n_tissues", "min_qval"]],
            how="left",
        )

    feats = feats.fillna(0.0)
    log.info("SNP feature matrix: %d SNPs x %d features", *feats.shape)
    return feats


def aggregate_block_features(
    snp_features: pd.DataFrame, snp_assignments: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate per-SNP features to per-block statistics.

    For each block we emit:
    - n_snps
    - mean and max of every numeric SNP feature
    Categorical / 0-1 indicators end up as proportions via the mean.
    """
    # Drop columns from snp_features that already live in snp_assignments to
    # avoid pandas suffix collisions (e.g. both carry `pos`).
    overlap = [c for c in snp_features.columns if c in snp_assignments.columns]
    feats = snp_features.drop(columns=overlap)
    df = snp_assignments.merge(feats, left_on="rsid", right_index=True, how="inner")
    feature_cols = list(feats.columns)
    grouped = df.groupby("block_id")
    means = grouped[feature_cols].mean().add_suffix("__mean")
    maxes = grouped[feature_cols].max().add_suffix("__max")
    counts = grouped.size().rename("n_snps").to_frame()
    out = counts.join(means).join(maxes)
    log.info("Block feature matrix: %d blocks x %d features", *out.shape)
    return out
