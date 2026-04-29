# CLEAR-A

Calibrated, LD-aware, Explainable Reduction for Association.

ML-based SNP search-space reduction for GWAS, validated on a real, fully-public openSNP-derived dataset. Course project for Advanced Bioinformatics, University of Moratuwa (258421E, 258423L).

## Data

The `opensnp.org` site is no longer running, but a processed, QC'd, imputed dump from Frei et al. is mirrored on Zenodo (DOI 10.5281/zenodo.10715132). It contains:

- 4,257 individuals × ~1.2M HapMap3 SNPs, hard-called from imputation against 1KGP Phase 3.
- Phenotype CSV with binary hair-color indicators (black / brown / blonde), height, handedness, sex, and PC1-PC10.
- Neale-lab UKB summary statistics for the same hair traits → independent cross-cohort for transferability.

The pipeline targets hair color as the primary phenotype; pigmentation has strong, well-characterised loci (MC1R, ASIP, HERC2/OCA2, KITLG, IRF4, TYR, SLC45A2) which makes "did your method recover known biology?" a defensible evaluation criterion.

## Pipeline

1. Download Zenodo dataset (`scripts/01_download_opensnp.py`).
2. Load into the project layout — pheno table + dosage matrix (`scripts/02_load_zenodo.py`).
3. Build LD blocks from LDetect + attach functional annotations (`scripts/04_build_blocks.py`).
4. Two-stage ranker: group-level (XGBoost on block features) → SNP-level (elastic net + DNN) (`scripts/05_train_clear_a.py`).
5. Bootstrap stability selection.
6. Post-selection FDR via knockoff filter (primary) + Westfall–Young permutation (comparator).
7. Per-SNP and per-block explanations (SHAP + annotation evidence).

See the proposal PDF at `../258421E, 258423L  Machine Learning Based Search Space Reduction in Genome-Wide Association Studies.pdf` and the plan file at `~/.claude/plans/dont-use-synthetic-data-zany-lecun.md`.

## Quickstart

```bash
pip install -e .[dev]

# 1.8 GB download. Use --insecure if your network does SSL inspection.
python scripts/01_download_opensnp.py --out data/raw_opensnp --insecure

python scripts/02_load_zenodo.py \
    --config configs/hair_blonde.yaml \
    --zenodo-dir data/raw_opensnp \
    --out-dir data/processed/hair_blonde

# Download LDetect EUR fourier_ls-all.bed once into data/reference_1000g/
python scripts/04_build_blocks.py \
    --config configs/hair_blonde.yaml \
    --data-dir data/processed/hair_blonde \
    --ldetect-bed data/reference_1000g/fourier_ls-all.bed \
    --out-dir data/processed/hair_blonde/blocks

python scripts/05_train_clear_a.py \
    --config configs/hair_blonde.yaml \
    --blocks-dir data/processed/hair_blonde/blocks \
    --phenotype-file data/processed/hair_blonde/hair_blonde.pheno \
    --gwas-catalog data/annotations/gwas_catalog_v1.0.2.tsv \
    --out-dir results/hair_blonde \
    --fast   # remove for full run
```

## Layout

- `src/clear_a/` — importable pipeline package.
- `scripts/` — CLI drivers for long-running stages.
- `notebooks/` — exploration and figure generation.
- `configs/` — per-phenotype YAML configs (hair_color multiclass, hair_{black,brown,blonde} binary).
- `data/`, `results/` — gitignored.

## External assets to download once

- LDetect EUR `fourier_ls-all.bed` — github.com/jmacdon/LDetectData
- LDSC `baselineLD_v2.2_annots/` (optional) — github.com/bulik/ldsc
- GWAS Catalog `gwas_catalog_v1.0.2-associations_*.tsv` — ebi.ac.uk/gwas/api/search/downloads
