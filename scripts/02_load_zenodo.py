"""Load the Zenodo openSNP dataset (pre-QC'd, pre-imputed) into the project layout.

Replaces the original `02_run_qc.py` for the Zenodo-sourced data: that
dataset already ships with QC + imputation done by Frei et al. We only need
to:
  1. Locate the imputed PLINK fileset (`imputed/opensnp_hm3.{bed,bim,fam}`).
  2. Read the phenotype CSV (`qc/pheno.csv`) and emit a CLEAR-A-format
     phenotype file for the requested trait.
  3. Read the genotype dosages into a numpy array for the downstream pipeline
     (no imputation server step needed).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from bed_reader import open_bed


# Map our config phenotype names to the columns in Zenodo's pheno.csv.
ZENODO_PHENOTYPE_MAP = {
    "hair_color": ["hair_black", "hair_brown", "hair_blonde"],
    "hair_black": ["hair_black"],
    "hair_brown": ["hair_brown"],
    "hair_blonde": ["hair_blonde"],
    "height": ["height_cm"],
    "lefthand": ["lefthand"],
    "ambidextrous": ["ambidextrous"],
}


def find_dataset_root(extracted_dir: Path) -> Path:
    """Locate the directory that contains qc/ and imputed/ subdirs."""
    if (extracted_dir / "qc" / "pheno.csv").exists():
        return extracted_dir
    for d in extracted_dir.iterdir():
        if d.is_dir() and (d / "qc" / "pheno.csv").exists():
            return d
    raise FileNotFoundError(f"Could not find qc/pheno.csv under {extracted_dir}")


def build_label_from_columns(pheno: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combine one or more binary indicator columns into a single integer label.

    For `hair_color` (three columns) we map black=0, brown=1, blonde=2.
    Subjects with multiple positive indicators are dropped (ambiguous).
    For a single column the values pass through (binary 0/1 or quantitative).
    """
    if len(columns) == 1:
        col = columns[0]
        s = pd.to_numeric(pheno[col], errors="coerce")
        return s
    # Multi-column case: enforce exactly-one-positive.
    sub = pheno[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    pos_count = (sub > 0).sum(axis=1)
    keep = pos_count == 1
    label_idx = sub.values.argmax(axis=1)
    out = pd.Series(np.where(keep, label_idx, np.nan), index=pheno.index, name="label")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--zenodo-dir", type=Path, required=True,
                        help="Directory containing the extracted Zenodo archive")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = find_dataset_root(args.zenodo_dir)
    pheno_path = root / "qc" / "pheno.csv"
    bed_prefix = root / "imputed" / "opensnp_hm3"
    logging.info("Dataset root: %s", root)

    # Despite the .csv extension, Frei et al.'s pheno table is tab-separated.
    pheno = pd.read_csv(pheno_path, sep="\t")
    logging.info("Loaded phenotype table: %d rows, columns=%s", len(pheno), list(pheno.columns))

    pheno_name = cfg["phenotype"]["name"]
    columns = ZENODO_PHENOTYPE_MAP.get(pheno_name)
    if columns is None:
        raise SystemExit(
            f"Phenotype {pheno_name!r} not in Zenodo dataset. Available: "
            f"{list(ZENODO_PHENOTYPE_MAP)}"
        )
    label = build_label_from_columns(pheno, columns)
    cleaned = pheno[["FID", "IID"]].copy()
    cleaned["label"] = label
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)
    pheno_out = args.out_dir / f"{pheno_name}.pheno"
    cleaned.to_csv(pheno_out, sep="\t", index=False)
    logging.info(
        "Wrote phenotype file %s with %d kept samples; label_counts=%s",
        pheno_out, len(cleaned),
        cleaned["label"].value_counts().sort_index().to_dict(),
    )

    logging.info("Reading PLINK fileset: %s", bed_prefix)
    bed = open_bed(f"{bed_prefix}.bed", count_A1=True)
    sample_ids = np.asarray(bed.iid)
    snp_info = pd.DataFrame({
        "rsid": np.asarray(bed.sid),
        "chrom": np.asarray(bed.chromosome),
        "pos": np.asarray(bed.bp_position).astype(int),
        "a0": np.asarray(bed.allele_1),  # A1 = minor / counted allele
        "a1": np.asarray(bed.allele_2),  # A2 = major / reference
    })
    n_samples, n_snps = bed.iid_count, bed.sid_count
    logging.info("Genotype matrix shape: %d samples x %d SNPs", n_samples, n_snps)

    # Memory-mapped int8 storage. Float32 (20 GB) does not fit in RAM; int8
    # (5 GB) is small enough on disk and downstream code can stream via
    # np.load(mmap_mode='r'). Missing genotypes become 0 (mean-impute is fine
    # because the data is post-imputation and missingness is rare).
    dosage_path = args.out_dir / "dosages.npy"
    out = np.lib.format.open_memmap(
        dosage_path, mode="w+", dtype="int8", shape=(n_samples, n_snps)
    )
    chunk = 25_000
    for i in range(0, n_snps, chunk):
        end = min(i + chunk, n_snps)
        block = bed.read(index=np.s_[:, i:end], dtype="int8")
        # bed-reader returns -127 for missing in int8 mode; replace with 0.
        block[block < 0] = 0
        out[:, i:end] = block
        if (i // chunk) % 10 == 0:
            logging.info("  read SNPs %d/%d", end, n_snps)
    out.flush()
    del out  # close mmap

    pd.DataFrame({"sample_id": sample_ids}).to_csv(args.out_dir / "samples.csv", index=False)
    snp_info.to_parquet(args.out_dir / "snp_info.parquet", index=False)
    logging.info(
        "Saved dosages (%d, %d) int8 to %s, %d samples, %d SNPs",
        n_samples, n_snps, dosage_path, len(sample_ids), len(snp_info),
    )


if __name__ == "__main__":
    main()
