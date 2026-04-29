"""Group SNPs into LD blocks using one of three definitions.

- `ldetect`: load precomputed Berisa-Pickrell EUR blocks (the de-facto default
  for European-ancestry GWAS). Distributed at github.com/jmacdon/LDetectData
  as `EUR/fourier_ls-all.bed`.
- `plink`: invoke `plink --blocks` on the supplied reference fileset.
- `bigld`: rectangular contiguous blocks via the BigLD R package (called via
  Rscript). Used as an ablation only; if R is unavailable we fall back.

All three produce a uniform `BlockTable` with one row per block:
  block_id, chrom, start_bp, end_bp, n_snps
plus a `SnpAssignment` table mapping rsid → block_id.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BlockResult:
    blocks: pd.DataFrame  # block_id, chrom, start_bp, end_bp, n_snps
    assignments: pd.DataFrame  # rsid, chrom, pos, block_id


def _make_block_id(chrom: str, start: int, end: int) -> str:
    return f"chr{chrom}:{start}-{end}"


def load_ldetect_blocks(ldetect_bed: Path) -> pd.DataFrame:
    """Read the LDetect Berisa-Pickrell BED into a tidy frame."""
    df = pd.read_csv(
        ldetect_bed,
        sep=r"\s+",
        comment="#",
        header=0,
        names=["chrom", "start_bp", "stop_bp"],
    )
    df["chrom"] = df["chrom"].str.replace("chr", "", regex=False)
    df["start_bp"] = df["start_bp"].astype(int)
    df["end_bp"] = df["stop_bp"].astype(int)
    df["block_id"] = [
        _make_block_id(c, s, e)
        for c, s, e in zip(df["chrom"], df["start_bp"], df["end_bp"])
    ]
    return df[["block_id", "chrom", "start_bp", "end_bp"]]


def assign_snps_to_blocks(snp_info: pd.DataFrame, blocks: pd.DataFrame) -> BlockResult:
    """Map each SNP (chrom, pos) to the block that contains it.

    Uses an interval merge per chromosome to stay O(N log N). SNPs in
    inter-block gaps (rare) get their own singleton block so nothing is lost.
    """
    out_assignments: list[pd.DataFrame] = []
    used_blocks: list[pd.DataFrame] = []

    for chrom, snp_chr in snp_info.groupby("chrom"):
        bk_chr = blocks[blocks["chrom"] == str(chrom)].sort_values("start_bp").reset_index(drop=True)
        snp_chr_sorted = snp_chr.sort_values("pos").reset_index(drop=True)
        if bk_chr.empty:
            singleton = snp_chr_sorted.copy()
            singleton["block_id"] = [
                _make_block_id(str(chrom), int(p), int(p)) for p in singleton["pos"]
            ]
            out_assignments.append(singleton[["rsid", "chrom", "pos", "block_id"]])
            used_blocks.append(
                pd.DataFrame({
                    "block_id": singleton["block_id"],
                    "chrom": str(chrom),
                    "start_bp": singleton["pos"],
                    "end_bp": singleton["pos"],
                })
            )
            continue

        # LDetect Berisa-Pickrell blocks use BED half-open intervals: block
        # N's stop equals block N+1's start, so closed="both" creates overlaps
        # that IntervalIndex.get_indexer rejects. Use left-closed to match BED.
        idx = pd.IntervalIndex.from_arrays(
            bk_chr["start_bp"], bk_chr["end_bp"], closed="left"
        )
        positions = snp_chr_sorted["pos"].to_numpy()
        loc = idx.get_indexer(positions)
        block_ids = []
        for pos, i in zip(positions, loc):
            if i == -1:
                block_ids.append(_make_block_id(str(chrom), int(pos), int(pos)))
            else:
                block_ids.append(bk_chr.loc[i, "block_id"])
        snp_chr_sorted = snp_chr_sorted.assign(block_id=block_ids)
        out_assignments.append(snp_chr_sorted[["rsid", "chrom", "pos", "block_id"]])
        used_blocks.append(bk_chr)

    assignments = pd.concat(out_assignments, ignore_index=True)
    n_per_block = assignments.groupby("block_id").size().rename("n_snps")
    blocks_out = (
        pd.concat(used_blocks, ignore_index=True)
        .drop_duplicates("block_id")
        .merge(n_per_block, left_on="block_id", right_index=True, how="left")
        .fillna({"n_snps": 0})
    )
    blocks_out["n_snps"] = blocks_out["n_snps"].astype(int)
    blocks_out = blocks_out[blocks_out["n_snps"] > 0].reset_index(drop=True)
    return BlockResult(blocks=blocks_out, assignments=assignments)


def plink_blocks(
    ref_prefix: Path, out_prefix: Path, plink2_bin: str | None = None
) -> pd.DataFrame:
    """Run PLINK 2 `--blocks` and parse the resulting `.blocks.det` file."""
    plink2 = plink2_bin or shutil.which("plink") or shutil.which("plink2")
    if not plink2:
        raise RuntimeError("plink not found on PATH (PLINK 2 --blocks requires PLINK 1.9 binary)")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_prefix.parent / "ld_logs" / f"{out_prefix.name}.blocks.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        subprocess.run(
            [plink2, "--bfile", str(ref_prefix), "--blocks", "no-pheno-req", "--out", str(out_prefix)],
            stdout=fh, stderr=subprocess.STDOUT, check=True,
        )
    det = pd.read_csv(f"{out_prefix}.blocks.det", sep=r"\s+")
    det = det.rename(columns={"CHR": "chrom", "BP1": "start_bp", "BP2": "end_bp"})
    det["chrom"] = det["chrom"].astype(str)
    det["block_id"] = [
        _make_block_id(c, s, e) for c, s, e in zip(det["chrom"], det["start_bp"], det["end_bp"])
    ]
    return det[["block_id", "chrom", "start_bp", "end_bp"]]


def build_blocks(
    snp_info: pd.DataFrame,
    method: str,
    ldetect_bed: Path | None = None,
    plink_ref_prefix: Path | None = None,
    plink_out_prefix: Path | None = None,
) -> BlockResult:
    """Top-level dispatch over block definitions."""
    method = method.lower()
    if method == "ldetect":
        if ldetect_bed is None:
            raise ValueError("ldetect requires ldetect_bed path")
        blocks = load_ldetect_blocks(ldetect_bed)
    elif method == "plink":
        if plink_ref_prefix is None or plink_out_prefix is None:
            raise ValueError("plink requires plink_ref_prefix and plink_out_prefix")
        blocks = plink_blocks(plink_ref_prefix, plink_out_prefix)
    else:
        raise ValueError(f"Unknown LD-block method: {method}")
    return assign_snps_to_blocks(snp_info, blocks)
