"""Build LD blocks and per-SNP / per-block feature matrices.

Consumes the artefacts produced by `scripts/02_load_zenodo.py`:
  - `dosages.npy` (samples x SNPs, float32, 0-2)
  - `samples.csv`
  - `snp_info.parquet` with columns rsid, chrom, pos, a0, a1

Produces, in `--out-dir`:
  - `snp_assignments.parquet` — rsid → block_id mapping
  - `blocks.parquet` — block_id, chrom, start_bp, end_bp, n_snps
  - `snp_features.parquet` — per-SNP numeric feature matrix
  - `block_features.parquet` — aggregated per-block features

LDSC baselineLD annotations are optional; without them the SNP feature matrix
falls back to position-only features (still functional, just less informative).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clear_a.annotations import (
    aggregate_block_features,
    build_snp_feature_matrix,
    load_ldsc_baseline,
)
from clear_a.ld_blocks import build_blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Output of 02_load_zenodo.py (contains dosages.npy etc.)")
    parser.add_argument("--ldetect-bed", type=Path, required=True)
    parser.add_argument("--ldsc-dir", type=Path, default=None,
                        help="Optional LDSC baselineLD annotation directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snp_info = pd.read_parquet(args.data_dir / "snp_info.parquet")
    logging.info("Loaded SNP info: %d variants", len(snp_info))

    block_result = build_blocks(
        snp_info,
        method=cfg["ld_blocks"]["method"],
        ldetect_bed=args.ldetect_bed,
    )
    block_result.assignments.to_parquet(args.out_dir / "snp_assignments.parquet", index=False)
    block_result.blocks.to_parquet(args.out_dir / "blocks.parquet", index=False)
    logging.info(
        "LD blocks: %d blocks covering %d SNPs",
        len(block_result.blocks), len(block_result.assignments),
    )

    ldsc = load_ldsc_baseline(args.ldsc_dir) if args.ldsc_dir else None
    snp_features = build_snp_feature_matrix(snp_info, ldsc_df=ldsc)
    snp_features.to_parquet(args.out_dir / "snp_features.parquet")

    block_features = aggregate_block_features(snp_features, block_result.assignments)
    block_features.to_parquet(args.out_dir / "block_features.parquet")

    # Pass through dosages and samples so 05_train_clear_a only needs --blocks-dir.
    import shutil
    for fname in ("dosages.npy", "samples.csv", "snp_info.parquet"):
        src = args.data_dir / fname
        dst = args.out_dir / fname
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    logging.info("Wrote LD blocks + features to %s", args.out_dir)


if __name__ == "__main__":
    main()
