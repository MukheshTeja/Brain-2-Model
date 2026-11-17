"""Verify the pinned download manifest and the actual model inputs, without downloading."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cdb2m.fmri.datasets import load_ds003076, load_hcp_relational, hcp_blocks


def audit(data_dir="data"):
    root = Path(data_dir)
    manifest_path = root / "MANIFEST.json"
    errors = []
    if not manifest_path.exists():
        return {"ok": False, "errors": ["Missing MANIFEST.json; run scripts.download_data"]}
    manifest = json.loads(manifest_path.read_text())
    for rel, meta in manifest["files"].items():
        p = root / rel
        if not p.is_file():
            errors.append(f"Missing: {rel}")
        elif p.stat().st_size != meta["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
            errors.append(f"Checksum/size mismatch: {rel}")
    result = {"ok": not errors, "errors": errors, "files_verified": len(manifest["files"]),
              "total_bytes": manifest["total_bytes"], "sources": manifest["sources"],
              "scope": "Released derivatives for the laptop pipeline; not full source-paper replications",
              "optional_not_required": ["raw BOLD and preprocessing", "B2M patient recordings", "ECoG podcast data", "ImageNet and human similarity judgments"]}
    # Never deserialize pickle inputs after a failed integrity check.
    if errors:
        return result
    try:
        ds = load_ds003076(root)
        assert len(ds.subjects) == 10 and len(ds.problems) == 70
        assert len(ds.betas) == 3
        for network in ds.betas.values():
            assert len(network) == 10
            assert all(x.shape[0] == 70 and x.shape[1] > 0 and np.isfinite(x).all() for x in network.values())
        h = load_hcp_relational(root)
        hb = hcp_blocks(h)
        assert len(h.betas) == 20 and len(hb) == 2
        assert all(p is not None for p in h.item_table.positions)
        assert all(np.isfinite(x).all() for b in hb.values() for x in b["betas"])
        result["ds003076"] = {"subjects": len(ds.subjects), "problems": len(ds.problems),
            "voxels_per_network": {k: sorted({x.shape[1] for x in v.values()}) for k, v in ds.betas.items()}}
        result["hcp"] = {"downloaded_subjects": len(h.betas), "usable_subjects": sum(len(b["subjects"]) for b in hb.values()),
                         "unique_stimuli": len(h.items), "blocks": {k: len(v["subjects"]) for k,v in hb.items()},
                         "exclusion": "One subject has a stimulus set without five matching subjects"}
    except Exception as exc:
        errors.append(f"Dataset load/shape validation failed: {type(exc).__name__}: {exc}")
    result["ok"] = not errors
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/laptop/data_audit.json")
    args = ap.parse_args()
    result = audit(args.data_dir)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
