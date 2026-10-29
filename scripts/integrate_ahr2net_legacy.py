#!/usr/bin/env python3
# coding: utf-8
"""Idempotent, backed-up registration of AHR2Net in the existing YOLO-tcy_v5 model factory.

Enables the original train_rlsfa_gtga_bhlr_v2.py and val_rlsfa_gtga_bhlr_v2.py
without replacing the user's existing training or evaluation implementation.
"""
from __future__ import annotations
import ast
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'models' / 'gtga_bhlr_factory.py'
BACKUP = ROOT / 'models' / 'gtga_bhlr_factory.before_ahr2net_legacy_integration.py'
MARK = '    # AHR2Net integrated dispatch: use original v5 trainer and validator\n'
OLD_MARK = '    # AHR2Net scratch integrated model dispatch (2026)\n'
NEEDLE = '    model_cfg = cfg.get("model", {}) or {}\n'

INJECT = '''    # AHR2Net integrated dispatch: use original v5 trainer and validator
    if str((cfg.get("model", {}) or {}).get("variant", "")).strip().lower() == "ahr2net_v5_scratch":
        from models.ahr2net_scratch_model import build_ahr2net_from_config
        ahr_cfg = dict(cfg)
        ahr_cfg["model"] = dict(cfg.get("model", {}) or {})
        if fusion_override is not None and str(fusion_override).lower() != "auto":
            ahr_cfg["model"]["fusion"] = str(fusion_override)
        if nc is not None:
            ahr_cfg["model"]["nc"] = int(nc)
        model = build_ahr2net_from_config(
            ahr_cfg, verbose=verbose, pretrained_override=pretrained,
        )
        if names is not None:
            model.names = names
        if hasattr(model, "args"):
            loss_cfg = cfg.get("loss", {}) or {}
            model.args.box = float(loss_cfg.get("box", 7.5))
            model.args.cls = float(loss_cfg.get("cls", 0.5))
            model.args.dfl = float(loss_cfg.get("dfl", 1.5))
            model.criterion = None
        return model

'''


def main():
    if not TARGET.is_file():
        raise FileNotFoundError(f'Missing factory: {TARGET}')
    if not (ROOT / 'models' / 'ahr2net_scratch_model.py').is_file():
        raise FileNotFoundError('AHR model source missing: models/ahr2net_scratch_model.py')

    src = TARGET.read_text(encoding='utf-8')
    if MARK in src:
        print('[SKIP] AHR2Net integrated in original factory already')
        return

    def_pos = src.find('def build_model_from_config(')
    if def_pos < 0:
        raise RuntimeError('Original factory builder not found. Nothing changed.')
    next_def = src.find('\ndef ', def_pos + 1)
    boundary = next_def if next_def >= 0 else len(src)
    location = src.find(NEEDLE, def_pos, boundary)
    if location < 0:
        raise RuntimeError('model_cfg assignment not found in builder. Nothing changed.')

    # Support either a pristine factory or the registration patch in the earlier AHR2Net zip.
    old_location = src.find(OLD_MARK, def_pos, location)
    if old_location >= 0:
        dst = src[:old_location] + INJECT + src[location:]
        print('[INFO] Replacing older AHR factory hook to apply loss weights and overrides')
    else:
        if 'ahr2net_v5_scratch' in src[def_pos:location]:
            raise RuntimeError('Unknown AHR factory hook found. Nothing changed.')
        dst = src[:location] + INJECT + src[location:]

    ast.parse(dst, filename=str(TARGET))
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(dst, encoding='utf-8')
    print('[OK] Integrated into:', TARGET)
    print('[BACKUP]', BACKUP)
    print('[NOTE] Original train/val scripts stay unchanged')


if __name__ == '__main__':
    main()
