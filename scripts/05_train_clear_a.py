"""End-to-end CLEAR-A training driver for one phenotype.

Reads the prebuilt artefacts from `scripts/04_build_blocks.py` plus the
phenotype file from `scripts/02_run_qc.py`, runs nested CV, stability
selection, FDR control, and writes the final `selected.parquet` per the
plan's Step 6.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from clear_a.evaluate import load_gwas_catalog_hits, recovery_curve
from clear_a.explain import assemble_selected_table
from clear_a.inference import knockoff_filter_per_block, westfall_young_permutation
from clear_a.models.group_ranker import GroupRankConfig, GroupRanker
from clear_a.models.snp_ranker import SnpRankConfig, SnpRanker
from clear_a.stability import StabilityConfig, bootstrap_selection, threshold_stable


def _align_samples(dosages: np.ndarray, sample_ids: np.ndarray, pheno: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pheno = pheno.set_index("IID")["label"].astype(int)
    common = pd.Index(sample_ids).intersection(pheno.index)
    keep_mask = pd.Index(sample_ids).isin(common)
    y = pheno.reindex(np.array(sample_ids)[keep_mask]).to_numpy()
    return dosages[keep_mask], y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blocks-dir", type=Path, required=True)
    parser.add_argument("--phenotype-file", type=Path, required=True)
    parser.add_argument("--gwas-catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fast", action="store_true", help="Smoke-test profile")
    parser.add_argument(
        "--chrom-subset", nargs="+", default=None,
        help="Restrict SNPs to these chromosomes (e.g. 16 for MC1R smoke test).",
    )
    parser.add_argument(
        "--top-m-blocks", type=int, default=None,
        help="Override config's top_m_blocks (block→SNP cutoff in the SnpRanker).",
    )
    parser.add_argument(
        "--bootstraps", type=int, default=None,
        help="Override config's stability.n_bootstraps.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snp_info = pd.read_parquet(args.blocks_dir / "snp_info.parquet")
    snp_assignments = pd.read_parquet(args.blocks_dir / "snp_assignments.parquet")
    block_features = pd.read_parquet(args.blocks_dir / "block_features.parquet")
    snp_features = pd.read_parquet(args.blocks_dir / "snp_features.parquet")
    dosages = np.load(args.blocks_dir / "dosages.npy", mmap_mode="r")  # int8 mmap
    sample_ids = pd.read_csv(args.blocks_dir / "samples.csv")["sample_id"].to_numpy()
    pheno_df = pd.read_csv(args.phenotype_file, sep="\t")

    dosages, y = _align_samples(dosages, sample_ids, pheno_df)
    logging.info("Aligned dataset: dosages=%s, y_unique=%s", dosages.shape, np.unique(y, return_counts=True))

    if args.chrom_subset:
        keep_chrom = set(str(c) for c in args.chrom_subset)
        snp_mask = snp_info["chrom"].astype(str).isin(keep_chrom).to_numpy()
        keep_cols = np.where(snp_mask)[0]
        dosages = np.asarray(dosages[:, keep_cols], dtype=np.float32)
        snp_info = snp_info.iloc[keep_cols].reset_index(drop=True)
        snp_assignments = snp_assignments[snp_assignments["rsid"].isin(snp_info["rsid"])].reset_index(drop=True)
        block_features = block_features.loc[block_features.index.isin(snp_assignments["block_id"])]
        snp_features = snp_features.loc[snp_features.index.isin(snp_info["rsid"])]
        logging.info("Restricted to chromosomes %s: %d SNPs, %d blocks",
                     sorted(keep_chrom), len(snp_info), len(block_features))
    else:
        # Materialise to float32 once. Caller is responsible for ensuring this fits.
        dosages = np.asarray(dosages, dtype=np.float32)

    rsids = snp_info["rsid"].tolist()
    seed = cfg["model"]["cv"]["seed"]

    grp_cfg = GroupRankConfig(
        n_estimators=100 if args.fast else cfg["model"]["group"]["n_estimators"],
        max_depth=cfg["model"]["group"]["max_depth"],
        learning_rate=cfg["model"]["group"]["learning_rate"],
        seed=seed,
    )
    snp_cfg = SnpRankConfig(
        elastic_net_alpha=cfg["model"]["snp"]["elastic_net_alpha"],
        dnn_hidden=tuple(cfg["model"]["snp"]["dnn_hidden"]),
        dnn_dropout=cfg["model"]["snp"]["dnn_dropout"],
        dnn_epochs=5 if args.fast else 30,
        seed=seed,
    )
    top_m = (
        args.top_m_blocks
        if args.top_m_blocks is not None
        else (10 if args.fast else cfg["model"]["snp"]["top_m_blocks"])
    )

    def fit_rank(idx: np.ndarray) -> pd.Series:
        d_sub, y_sub = dosages[idx], y[idx]
        grp = GroupRanker(grp_cfg).fit(block_features, d_sub, y_sub, snp_assignments)
        block_scores = grp.rank(block_features)
        top_blocks = block_scores.head(top_m).index.tolist()
        keep_rsids = snp_assignments[snp_assignments["block_id"].isin(top_blocks)]["rsid"].tolist()
        col_idx = [i for i, r in enumerate(rsids) if r in set(keep_rsids)]
        if not col_idx:
            return pd.Series(dtype=float)
        sub_dos = d_sub[:, col_idx]
        sub_names = [rsids[i] for i in col_idx]
        snp = SnpRanker(snp_cfg).fit(sub_dos, snp_features.loc[sub_names].to_numpy(), y_sub, sub_names)
        return snp.rank(sub_dos)

    stab_cfg = StabilityConfig(
        n_bootstraps=(
            args.bootstraps if args.bootstraps is not None
            else (10 if args.fast else cfg["stability"]["n_bootstraps"])
        ),
        top_k=200 if args.fast else 1000,
        seed=seed,
    )
    pi_snps = bootstrap_selection(fit_rank, n_samples=len(y), config=stab_cfg)

    # Block-level stability via the same bootstraps but on block scores.
    def fit_block_rank(idx: np.ndarray) -> pd.Series:
        grp = GroupRanker(grp_cfg).fit(block_features, dosages[idx], y[idx], snp_assignments)
        return grp.rank(block_features)
    pi_blocks = bootstrap_selection(
        fit_block_rank, n_samples=len(y), config=stab_cfg, top_k_override=top_m,
    )

    stable = threshold_stable(
        pi_blocks, pi_snps,
        cfg["stability"]["pi_block_threshold"],
        cfg["stability"]["pi_snp_threshold"],
    )
    stable.to_parquet(args.out_dir / "stable_selection.parquet", index=False)

    selected_idx = [i for i, r in enumerate(rsids) if r in set(stable["rsid"])]
    if len(selected_idx) < 5:
        # Strict stability filter is empty — fall back to top 500 by selection frequency
        # so the FDR step still has a candidate pool to test.
        fallback = pi_snps.nlargest(500).index.tolist()
        logging.warning(
            "Stable SNP set has %d entries; falling back to top %d by π for inference.",
            len(selected_idx), len(fallback),
        )
        selected_idx = [i for i, r in enumerate(rsids) if r in set(fallback)]
    sub_dos = dosages[:, selected_idx]
    sub_rsids = [rsids[i] for i in selected_idx]
    if cfg["inference"]["primary"] == "knockoff" and len(selected_idx) >= 5:
        infer = knockoff_filter_per_block(
            sub_dos, y, sub_rsids,
            snp_assignments[snp_assignments["rsid"].isin(sub_rsids)],
            fdr=cfg["inference"]["fdr_target"], seed=seed,
        )
    else:
        infer = westfall_young_permutation(
            sub_dos, y, sub_rsids,
            n_perms=100 if args.fast else cfg["inference"]["permutation"]["n_perms"],
            fdr=cfg["inference"]["fdr_target"],
        )
    infer.table.to_parquet(args.out_dir / "inference.parquet", index=False)

    catalog_hits = load_gwas_catalog_hits(
        args.gwas_catalog, cfg["evaluation"]["gwas_catalog_traits"]
    )
    rec = recovery_curve(
        pi_snps.sort_values(ascending=False).reset_index()["index"],
        catalog_hits,
        cfg["evaluation"]["top_k_curve"],
    )
    rec.to_csv(args.out_dir / "recovery_curve.csv", index=False)

    selected_df = infer.table[infer.table["selected"]][["rsid"]]
    snp_scores = pi_snps.copy().rename("snp_score")
    assemble_selected_table(
        selected_df, snp_assignments, pi_blocks, pi_snps, snp_scores,
        snp_features, catalog_hits,
        args.out_dir / "selected.parquet",
    )
    logging.info("CLEAR-A pipeline complete. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
