#!/usr/bin/env python3
"""Check whether BHLR Dual-HR variants are configured correctly; optional real forward."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "control": None, "clean": "clean", "dual": "dual", "guided": "guided"
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="Build real model and run one inference on CUDA")
    ap.add_argument("--device", default="5")
    ap.add_argument("--semantic-imgsz", type=int, default=256)
    ap.add_argument("--mode", default="guided", choices=CASES.keys())
    args = ap.parse_args()
    assert (ROOT / "models/modules/bhlr_dual_hr.py").is_file()
    factory = (ROOT / "models/gtga_bhlr_factory.py").read_text()
    assert factory.count("# BHLR_DUAL_HR_V1_FACTORY_INTEGRATION") == 1
    for label, expected in CASES.items():
        path = ROOT / f"configs/experiments/bhlr_dual_hr/uavcb_hvad_{label}_officialPT_100e.yaml"
        cfg = yaml.safe_load(path.read_text())
        model = cfg["model"]
        assert model["name"] == "yolo26m-p2"
        assert model["bhlr"]["enabled"]
        assert model["hvad"]["enabled"]
        assert model["pretrained"]
        assert cfg["train"]["resume"] is None
        assert not cfg["train"]["stage2"]["enabled"]
        assert cfg["train"]["epochs"] == 100
        dc = model.get("bhlr_dual_hr", {})
        assert (dc.get("mode") if dc.get("enabled", False) else None) == expected
        print("[PASS]", label, path.relative_to(ROOT))
    original = yaml.safe_load((ROOT / "configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml").read_text())
    assert not original["model"].get("bhlr_dual_hr", {}).get("enabled", False)
    assert not original["model"].get("hvad", {}).get("enabled", False)
    print("[PASS] original GMLS-v2 remains independent")
    if not args.build:
        print("[OK] Static checks complete. Use --build for one real GPU forward.")
        return
    import torch
    import sys
    sys.path.insert(0, str(ROOT))
    from models.gtga_bhlr_factory import build_model_from_config
    cfg_file = ROOT / f"configs/experiments/bhlr_dual_hr/uavcb_hvad_{args.mode}_officialPT_100e.yaml"
    cfg = yaml.safe_load(cfg_file.read_text())
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg, pretrained=False, verbose=True).to(device).eval()
    s = args.semantic_imgsz
    high = torch.rand(1, 3, s * 3, s * 3, device=device)
    low = torch.rand(1, 3, s, s, device=device)
    tir = torch.rand(1, 3, s, s, device=device)
    with torch.inference_mode():
        x = model(high, tir, rgb_semantic=low, return_features=True)
    print("[PASS] real forward", args.mode, type(x), "on", device)


if __name__ == "__main__":
    main()
