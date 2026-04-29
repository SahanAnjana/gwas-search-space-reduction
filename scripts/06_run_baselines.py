"""Baseline GWAS / ML methods on the same QC'd, imputed dataset.

For fair comparison with CLEAR-A, every baseline consumes the same `dosages`,
`snp_info`, and phenotype split. Outputs go to `out_dir/<method>/ranking.csv`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def plain_gwas_chisq(dosages: np.ndarray, y: np.ndarray, rsids: list[str]) -> pd.DataFrame:
    carriers = (dosages > 0.5).astype(np.int8)
    y_bin = (y > 0).astype(np.int8)
    pvals = np.empty(dosages.shape[1], dtype=np.float32)
    for j in range(dosages.shape[1]):
        c = carriers[:, j]
        tab = np.array([
            [((c == 1) & (y_bin == 1)).sum(), ((c == 1) & (y_bin == 0)).sum()],
            [((c == 0) & (y_bin == 1)).sum(), ((c == 0) & (y_bin == 0)).sum()],
        ])
        if tab.min() == 0:
            pvals[j] = 1.0
        else:
            _, p, _, _ = stats.chi2_contingency(tab, correction=False)
            pvals[j] = p
    return pd.DataFrame({"rsid": rsids, "score": -np.log10(np.clip(pvals, 1e-300, 1.0)), "pvalue": pvals})


def lasso_logistic(dosages: np.ndarray, y: np.ndarray, rsids: list[str], seed: int = 42) -> pd.DataFrame:
    scaler = StandardScaler()
    X = scaler.fit_transform(dosages.astype(np.float32))
    model = LogisticRegression(
        penalty="l1", solver="saga", C=0.1, max_iter=2000, random_state=seed, n_jobs=-1,
    )
    model.fit(X, y)
    coefs = np.abs(model.coef_[0]) if model.coef_.ndim == 2 and model.coef_.shape[0] == 1 else np.abs(model.coef_).max(axis=0)
    return pd.DataFrame({"rsid": rsids, "score": coefs})


def random_forest(dosages: np.ndarray, y: np.ndarray, rsids: list[str], seed: int = 42) -> pd.DataFrame:
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)
    rf.fit(dosages.astype(np.float32), y)
    return pd.DataFrame({"rsid": rsids, "score": rf.feature_importances_})


def combi_linear_svm(dosages: np.ndarray, y: np.ndarray, rsids: list[str]) -> pd.DataFrame:
    """COMBI step 1: chromosome-wise linear SVM weights as relevance scores."""
    from sklearn.svm import LinearSVC
    scaler = StandardScaler()
    X = scaler.fit_transform(dosages.astype(np.float32))
    svm = LinearSVC(C=0.1, max_iter=5000)
    svm.fit(X, y)
    coefs = np.abs(svm.coef_[0]) if svm.coef_.ndim == 2 else np.abs(svm.coef_)
    return pd.DataFrame({"rsid": rsids, "score": coefs.ravel()[: len(rsids)]})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-dir", type=Path, required=True)
    parser.add_argument("--phenotype-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+",
                        default=["gwas_chisq", "lasso", "rf", "combi"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snp_info = pd.read_parquet(args.blocks_dir / "snp_info.parquet")
    dosages = np.load(args.blocks_dir / "dosages.npy", mmap_mode="r")  # int8 mmap
    sample_ids = pd.read_csv(args.blocks_dir / "samples.csv")["sample_id"].to_numpy()
    pheno = pd.read_csv(args.phenotype_file, sep="\t").set_index("IID")["label"].astype(int)
    keep = pd.Index(sample_ids).isin(pheno.index)
    y = pheno.reindex(np.array(sample_ids)[keep]).to_numpy()
    dosages = dosages[keep]
    rsids = snp_info["rsid"].tolist()

    runners = {
        "gwas_chisq": lambda: plain_gwas_chisq(dosages, y, rsids),
        "lasso": lambda: lasso_logistic(dosages, y, rsids),
        "rf": lambda: random_forest(dosages, y, rsids),
        "combi": lambda: combi_linear_svm(dosages, y, rsids),
    }
    for m in args.methods:
        if m not in runners:
            logging.warning("Unknown method: %s", m)
            continue
        out = runners[m]()
        out_path = args.out_dir / m / "ranking.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.sort_values("score", ascending=False).to_csv(out_path, index=False)
        logging.info("Wrote %s ranking → %s", m, out_path)


if __name__ == "__main__":
    main()
