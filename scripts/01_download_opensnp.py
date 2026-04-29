"""Download the processed openSNP dataset from Zenodo.

opensnp.org is no longer live (the domain is parked on DreamHost). The
canonical replacement is the Zenodo deposit 10.5281/zenodo.10715132 by
Frei et al., which mirrors the full openSNP genotype dump (Feb 2024) plus:

- pre-QC'd PLINK filesets split into chip-version clusters
- imputed hard calls against 1KGP HapMap3 (4,257 individuals, 1.2M SNPs)
- a phenotype CSV with hair color (binary), height, handedness, sex, PC1-PC10
- Neale-lab UKB summary statistics for the same hair-color traits — useful as
  an independent, higher-N cross-cohort for transferability checks

This script downloads the 1.8 GB tarball, extracts it, and verifies the
expected files are present.
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
import warnings
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "10715132"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
EXPECTED_FILES = (
    "qc/pheno.csv",
    "imputed/opensnp_hm3.bed",
    "imputed/opensnp_hm3.bim",
    "imputed/opensnp_hm3.fam",
)


def _ssl_verify_setting(insecure: bool, ca_bundle: Path | None) -> bool | str:
    if insecure:
        warnings.filterwarnings(
            "ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning
        )
        return False
    if ca_bundle is not None:
        return str(ca_bundle)
    env_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if env_bundle:
        return env_bundle
    return True


def stream_download(url: str, dest: Path, resume: bool = True, verify: bool | str = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    mode = "wb"
    pos = 0
    if resume and dest.exists():
        pos = dest.stat().st_size
        headers["Range"] = f"bytes={pos}-"
        mode = "ab"
    with requests.get(url, stream=True, headers=headers, timeout=60, verify=verify) as r:
        if r.status_code == 416:
            logging.info("File already fully downloaded: %s", dest)
            return dest
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + pos
        with open(dest, mode) as fh, tqdm(
            total=total, initial=pos, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if not chunk:
                    continue
                fh.write(chunk)
                pbar.update(len(chunk))
    return dest


def fetch_record_files(verify: bool | str) -> list[dict]:
    r = requests.get(ZENODO_API, timeout=60, verify=verify)
    r.raise_for_status()
    return r.json().get("files", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download processed openSNP from Zenodo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--insecure", action="store_true",
        help="Disable TLS certificate verification (corporate SSL inspection workaround).",
    )
    parser.add_argument("--ca-bundle", type=Path, default=None)
    parser.add_argument(
        "--filename", default="opensnp_2024_02_27.tar.gz",
        help="Zenodo file to download (default: full processed dataset, 1.8 GB).",
    )
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    verify = _ssl_verify_setting(args.insecure, args.ca_bundle)
    if verify is False:
        logging.warning("TLS verification disabled (--insecure).")

    args.out.mkdir(parents=True, exist_ok=True)
    files = fetch_record_files(verify)
    target = next((f for f in files if f["key"] == args.filename), None)
    if target is None:
        available = ", ".join(f["key"] for f in files)
        raise SystemExit(f"File {args.filename!r} not in Zenodo record. Available: {available}")

    url = target["links"]["self"]
    dest = args.out / target["key"]
    logging.info("Downloading %s (%.1f MB) → %s", target["key"], target["size"] / 1e6, dest)
    stream_download(url, dest, verify=verify)

    if args.skip_extract:
        return
    logging.info("Extracting %s", dest)
    with tarfile.open(dest, "r:gz") as tf:
        tf.extractall(args.out)

    missing = [p for p in EXPECTED_FILES if not (args.out / p).exists()]
    if missing:
        # Some archives nest under a top-level directory; locate it dynamically.
        candidates = [d for d in args.out.iterdir() if d.is_dir()]
        for c in candidates:
            if (c / "qc" / "pheno.csv").exists():
                logging.info("Dataset root: %s", c)
                return
        raise SystemExit(f"Missing expected files after extract: {missing}")
    logging.info("Extraction OK; expected files present.")


if __name__ == "__main__":
    main()
