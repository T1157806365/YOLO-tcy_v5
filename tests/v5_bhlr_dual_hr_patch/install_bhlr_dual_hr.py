#!/usr/bin/env python3
"""Safely install optional BHLR Dual-HR into an existing YOLO-tcy_v5 checkout.

Only factory registration and NEW module/YAML/check scripts are added.
No modification to BHLR-v2/GTGA/HVAD/YOLO training core or original GMLS config.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from datetime import datetime

import yaml

SENTINEL = "# BHLR_DUAL_HR_V1_FACTORY_INTEGRATION"
FACTORY_BLOCK = '''    # BHLR_DUAL_HR_V1_FACTORY_INTEGRATION
    if bool((model_cfg.get("bhlr_dual_hr", {}) or {}).get("enabled", False)):
        from models.modules.bhlr_dual_hr import attach_dual_hr
        model = attach_dual_hr(model, model_cfg["bhlr_dual_hr"], verbose=verbose)

'''
EXPECTED_ANCHOR = '''    # HVAD_GEOHR_V1_FACTORY_INTEGRATION
    if bool((model_cfg.get("hvad", {}) or {}).get("enabled", False)):
        from hvad.runtime import attach_hvad
        model = attach_hvad(model, model_cfg["hvad"], verbose=verbose)

    return model
'''
BASE_YAML = (
    "configs/experiments/hvad_v1/"
    "uavcb_yolo26m_p2_gtga_bhlr_tri_gate_hvad_100e.yaml"
)


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    p = path.with_name(path.name + ".before_dualhr_" + stamp)
    shutil.copy2(path, p)
    return p


def write_if_changed(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        print("[unchanged]", path)
        return
    if path.exists():
        print("[backup]", backup(path))
    path.write_bytes(data)
    print("[write]", path)


def create_variants(target: Path):
    source = target / BASE_YAML
    if not source.is_file():
        raise FileNotFoundError(f"Cannot find source configuration: {source}")
    base = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or not isinstance(base.get("model"), dict):
        raise ValueError("Source HVAD YAML does not contain model dictionary")
    import copy

    variants = [
        ("control", None, "yolo26m_UAVCB_P2_HVAD_Control_DualHR_100e_seed0"),
        ("clean", "clean", "yolo26m_UAVCB_P2_HVAD_CleanResidual_100e_seed0"),
        ("dual", "dual", "yolo26m_UAVCB_P2_HVAD_DualHR_100e_seed0"),
        ("guided", "guided", "yolo26m_UAVCB_P2_HVAD_GuidedDualHR_100e_seed0"),
    ]
    out = target / "configs/experiments/bhlr_dual_hr"
    out.mkdir(parents=True, exist_ok=True)
    for label, mode, name in variants:
        cfg = copy.deepcopy(base)
        cfg.pop("traincc", None)  # stale command block in original YAML
        cfg["model"].pop("bhlr_dual_hr", None)
        if mode is not None:
            cfg["model"]["bhlr_dual_hr"] = {
                "enabled": True,
                "mode": mode,
                "detail_channels": int(cfg["model"]["bhlr"].get("detail_channels", 48)),
                "stem_channels": int(cfg["model"]["bhlr"].get("stem_channels", 16)),
                "ratio": int(cfg["model"]["bhlr"].get("high_to_semantic_ratio", 3)),
                "gate_hidden": 24,
                "max_gain": 0.50,
                "detach_guidance": True,
                "min_guided_prior": 0.15,
            }
        cfg["model"]["pretrained"] = True
        tr = cfg.setdefault("train", {})
        tr.update({
            "device": "5,6",
            "batch": 24,
            "batch_mode": "global",
            "epochs": 100,
            "resume": None,
            "ddp_find_unused_parameters": True,
            "seed": 0,
            "amp": True,
        })
        tr.setdefault("stage2", {})["enabled"] = False
        tr["stage2"]["epochs"] = 0
        cfg["output"] = {"project": "runs/bhlr_dual_hr", "name": name}
        if cfg["model"].get("hvad", {}).get("enabled", False) is not True:
            raise ValueError("This generator expects the existing HVAD baseline")
        cfg["experiment_note"] = (
            "BHLR Dual-HR controlled experiment. Independent official-YOLO-pretrained "
            "100e run, not a finetune from an old best.pt."
        )
        out_file = out / f"uavcb_hvad_{label}_officialPT_100e.yaml"
        header = (
            "# YOLO-tcy_v5 / backup-20261117-no-teacher\n"
            f"# BHLR Dual-HR study: {label}; start from official YOLO pretrained, no teacher.\n"
            "# Original GMLS-v2 and HVAD YAML are not modified.\n"
        )
        payload = header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=100)
        write_if_changed(out_file, payload.encode("utf-8"))
        print("[variant]", label, out_file.relative_to(target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path, help="Existing YOLO-tcy_v5 root")
    args = ap.parse_args()
    target = args.target.expanduser().resolve()
    this = Path(__file__).resolve().parent
    factory = target / "models/gtga_bhlr_factory.py"
    bhlr = target / "models/modules/bhlr_v2.py"
    hmod = target / "hvad/runtime.py"
    if not (factory.is_file() and bhlr.is_file() and hmod.is_file()):
        raise FileNotFoundError("Target does not look like the expected v5 HVAD repository")
    factory_text = factory.read_text(encoding="utf-8")
    if SENTINEL not in factory_text:
        if factory_text.count(EXPECTED_ANCHOR) != 1:
            raise RuntimeError(
                "Factory anchor changed: will not modify factory blindly. "
                "Please provide models/gtga_bhlr_factory.py."
            )
        new = factory_text.replace(
            EXPECTED_ANCHOR,
            EXPECTED_ANCHOR.replace("    return model\n", FACTORY_BLOCK + "    return model\n"),
        )
        write_if_changed(factory, new.encode("utf-8"))
    else:
        print("[unchanged] factory integration present")

    write_if_changed(
        target / "models/modules/bhlr_dual_hr.py",
        (this / "models/modules/bhlr_dual_hr.py").read_bytes(),
    )
    for filename in ("check_bhlr_dual_hr.py", "diag_hr_residual.py"):
        write_if_changed(
            target / "scripts" / filename,
            (this / "scripts" / filename).read_bytes(),
        )
    create_variants(target)
    print("\n[OK] Optional Dual-HR installed. Original gmls_v2 YAML unchanged.")
    print("First: python scripts/check_bhlr_dual_hr.py")


if __name__ == "__main__":
    main()
