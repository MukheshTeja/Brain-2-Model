"""Download every dataset CD-B2M needs into ./data (about 20 MB total).

Usage:
    python -m scripts.download_data                 # default, ~20 MB, a few seconds
    python -m scripts.download_data --with-raw-bold # optional: + raw ds003076 BOLD/T1w
                                                    #   for the 10 analysed subjects
                                                    #   (~1.3 GB, needs OpenNeuro/S3 access)

What is downloaded (all pinned to exact commits so results are reproducible):

1. ds003076 "Brain correlates of deductive reasoning in adults" (OpenNeuro,
   CC0; Lytle, Prado & Booth 2020) -- trial-wise fMRI responses for the 72
   reasoning problems x 10 subjects, already preprocessed (SPM12 + GLMsingle,
   top-10% task-responsive voxels) and released by Xiao et al. 2026 (Nat.
   Mach. Intell., MIT licence, github.com/pkuxmq/Brain-guided_LLM). Three ROI
   families: deductive-reasoning network (Prado 2011 meta-analysis spheres),
   multiple-demand network, language network.
2. HCP relational-processing task fMRI -- trial-wise responses for 24 HCP
   subjects in the MD network (same release), plus stimulus->problem mapping.
3. ds003076 BIDS metadata (events.tsv with the exact premises/conclusions,
   participants.tsv, dataset_description.json) from OpenNeuro's official
   GitHub mirror (github.com/OpenNeuroDatasets/ds003076).
4. ROI masks (MNI NIfTI) and the Prado 2011 peak coordinates used to build
   the deductive-reasoning ROI -- needed only for the optional raw-BOLD path.
5. Xiao et al.'s generated deductive-reasoning test sets (3/4/5/6 premises,
   propositional) -- used as an independent cross-check of our own generator.

Why preprocessed betas instead of raw BOLD: the raw dataset is ~1.3 GB for
the 10 subjects (more for all 17) and needs SPM/fMRIPrep-scale
preprocessing (hours, several GB of RAM). The released betas are the exact
inputs Xiao et al.'s published analyses used, are 13 MB, and load in
milliseconds. This is the "minimal compute" path.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

XIAO_REPO = "pkuxmq/Brain-guided_LLM"
XIAO_SHA = "f4c6f6b1b737a3885e198185456c7f1241fd2149"
DS_REPO = "OpenNeuroDatasets/ds003076"
DS_SHA = "aa412641e9240e0b35915a90db933e6d6c6cbeb9"

# (path prefix inside Xiao repo) -> (destination under data/)
XIAO_PATHS = {
    "fmri_data/preprocessed_data_glmsinglesep_newdrroi_topksep/top-10%/": "ds003076_betas/",
    "neuroimaging_info/": "ds003076_info/",
    "hcp_fmri_results/stimdurMeanRT/": "hcp_relational/betas/",
    "stimuli_mapping_final.jsonl": "hcp_relational/stimuli_mapping_final.jsonl",
    "data/relational_test_set.jsonl": "hcp_relational/relational_test_set.jsonl",
    "data/relational_val_set.jsonl": "hcp_relational/relational_val_set.jsonl",
    "data/deductive_reasoning_data_test": "xiao_generated_sets/deductive_reasoning_data_test",
    "matlab/roi/": "rois/",
    "matlab/run_get_dr_roi.m": "rois/run_get_dr_roi.m",
    "LICENSE": "LICENSE_xiao_et_al_MIT.txt",
}

ANALYSED_SUBJECTS = ["sub-1001", "sub-1004", "sub-1008", "sub-1014", "sub-1017",
                     "sub-1019", "sub-1022", "sub-1027", "sub-1030", "sub-1034"]
RUNS = ["Syllogisms_run-01", "Syllogisms_run-02", "Transitive_run-01", "Transitive_run-02"]


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cd-b2m-downloader"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _repo_files(repo: str, sha: str) -> dict[str, bytes]:
    """Return {relative_path: bytes} for every regular file in repo@sha.
    Tries the GitHub zip archive first (pure Python), then `git clone`."""
    url = f"https://codeload.github.com/{repo}/zip/{sha}"
    try:
        blob = _fetch(url)
        out = {}
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # symlinks (git-annex pointers) are stored with the S_IFLNK bit
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                rel = info.filename.split("/", 1)[1]
                out[rel] = zf.read(info)
        print(f"  fetched {repo}@{sha[:8]} via zip archive ({len(blob) / 1e6:.1f} MB)")
        return out
    except Exception as exc:  # noqa: BLE001 - fall back to git on any network/zip failure
        print(f"  zip archive unavailable ({exc}); falling back to git clone")
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run(["git", "clone", "--quiet", f"https://github.com/{repo}", str(tmp / "r")], check=True)
        subprocess.run(["git", "-C", str(tmp / "r"), "checkout", "--quiet", sha], check=True)
        out = {}
        for p in (tmp / "r").rglob("*"):
            if p.is_file() and not p.is_symlink() and ".git" not in p.parts:
                out[p.relative_to(tmp / "r").as_posix()] = p.read_bytes()
        print(f"  fetched {repo}@{sha[:8]} via git clone")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def download_xiao(data_dir: Path) -> int:
    files = _repo_files(XIAO_REPO, XIAO_SHA)
    n = 0
    for rel, blob in files.items():
        for src_prefix, dst_prefix in XIAO_PATHS.items():
            if rel.startswith(src_prefix):
                tail = rel[len(src_prefix):]
                dst = dst_prefix + tail if dst_prefix.endswith("/") else dst_prefix + tail
                _write(data_dir / dst, blob)
                n += 1
                break
    return n


def download_bids_metadata(data_dir: Path) -> int:
    files = _repo_files(DS_REPO, DS_SHA)
    n = 0
    for rel, blob in files.items():
        if rel.endswith((".tsv", ".json")) or rel in ("CHANGES", "README"):
            _write(data_dir / "ds003076_bids" / rel, blob)
            n += 1
    return n


def download_raw_bold(data_dir: Path) -> None:
    """Optional: raw NIfTI for the 10 analysed subjects (OpenNeuro S3, public)."""
    base = "https://s3.amazonaws.com/openneuro.org/ds003076"
    for sub in ANALYSED_SUBJECTS:
        paths = [f"{sub}/anat/{sub}_T1w.nii.gz"] + [f"{sub}/func/{sub}_task-{r}_bold.nii.gz" for r in RUNS]
        for rel in paths:
            dest = data_dir / "ds003076_bids" / rel
            if dest.exists() and dest.stat().st_size > 0:
                continue
            print(f"  {rel}")
            _write(dest, _fetch(f"{base}/{rel}", timeout=600))


def write_manifest(data_dir: Path) -> None:
    manifest = {}
    total = 0
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            blob = p.read_bytes()
            total += len(blob)
            manifest[p.relative_to(data_dir).as_posix()] = {
                "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    meta = {"sources": {XIAO_REPO: XIAO_SHA, DS_REPO: DS_SHA}, "total_bytes": total, "files": manifest}
    (data_dir / "MANIFEST.json").write_text(json.dumps(meta, indent=1))
    print(f"  {len(manifest)} files, {total / 1e6:.1f} MB -> {data_dir / 'MANIFEST.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--with-raw-bold", action="store_true")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] ds003076 betas + HCP relational betas + ROIs (Xiao et al. 2026 release)")
    print(f"  wrote {download_xiao(data_dir)} files")
    print("[2/3] ds003076 BIDS metadata/events (OpenNeuro GitHub mirror)")
    print(f"  wrote {download_bids_metadata(data_dir)} files")
    if args.with_raw_bold:
        print("[3/3] raw ds003076 BOLD + T1w for the 10 analysed subjects (OpenNeuro S3)")
        try:
            download_raw_bold(data_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  raw download failed: {exc}\n  (needs unrestricted access to s3.amazonaws.com)")
            sys.exit(1)
    else:
        print("[3/3] raw BOLD skipped (pass --with-raw-bold for the optional fine-grained RQ2 path)")
    write_manifest(data_dir)


if __name__ == "__main__":
    main()
