from __future__ import annotations

from pathlib import Path

import pandas as pd

from clear_a.ld_blocks import assign_snps_to_blocks, load_ldetect_blocks


def test_assign_snps_to_blocks_handles_gaps() -> None:
    snps = pd.DataFrame({
        "rsid": ["rsA", "rsB", "rsC", "rsD"],
        "chrom": ["1", "1", "1", "2"],
        "pos": [500, 1500, 4000, 100],
    })
    blocks = pd.DataFrame({
        "block_id": ["chr1:1000-2000", "chr1:3000-5000"],
        "chrom": ["1", "1"],
        "start_bp": [1000, 3000],
        "end_bp": [2000, 5000],
    })
    out = assign_snps_to_blocks(snps, blocks)
    assignments = out.assignments.set_index("rsid")["block_id"]
    assert assignments["rsB"] == "chr1:1000-2000"
    assert assignments["rsC"] == "chr1:3000-5000"
    # rsA falls before any block → singleton; rsD has no chr2 block → singleton.
    assert assignments["rsA"].startswith("chr1:500-500")
    assert assignments["rsD"].startswith("chr2:100-100")
    assert (out.blocks["n_snps"] > 0).all()


def test_load_ldetect_blocks(tmp_path: Path) -> None:
    bed = tmp_path / "fourier.bed"
    bed.write_text("chr start stop\nchr1 1 1000\nchr2 5 50\n")
    blocks = load_ldetect_blocks(bed)
    assert list(blocks["chrom"]) == ["1", "2"]
    assert list(blocks["start_bp"]) == [1, 5]
    assert list(blocks["end_bp"]) == [1000, 50]
    assert all(bid.startswith("chr") for bid in blocks["block_id"])
