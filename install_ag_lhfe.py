#!/usr/bin/env python3
"""Minimal, version-checked factory adapter for YOLO-tcy_v5/main.

Keeps existing variants and model files intact. Backs up the only modified
source file. Idempotent; --check does not modify anything.
"""
from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path.cwd()
FACTORY = PROJECT / 'models/gtga_bhlr_factory.py'
BACKUP = PROJECT / 'models/gtga_bhlr_factory.py.before_ag_lhfe_v1'
MARK = '    elif variant in {\n        "rlsfa_gtga_centered_bhlr_v2",'
TAG = 'elif variant == "rlsfa_gtga_centered_ag_lhfe_v1"'

BLOCK = '''    elif variant == "rlsfa_gtga_centered_ag_lhfe_v1":
        from ag_lhfe.model import RGBTAGLHFEDetectionModel
        align_cfg = dict(model_cfg.get("rlsfa_gtga_centered", model_cfg.get("alignment", {})) or {})
        align_loss_cfg = dict(loss_cfg.get("alignment", loss_cfg.get("rlsfa_gtga_centered", {})) or {})
        model = RGBTAGLHFEDetectionModel(
            model_name=str(model_cfg.get("name", "yolo26m")),
            nc=int(nc),
            pretrained=bool(pretrained),
            fusion=fusion,
            fusion_indices=fusion_indices,
            align_mode=str(model_cfg.get("align_mode", "bilinear")),
            names=names,
            semantic_imgsz=int(data_cfg.get("rgb_semantic_imgsz", 640)),
            rlsfa_gtga_centered_cfg=align_cfg,
            rlsfa_gtga_centered_loss_cfg=align_loss_cfg,
            ag_lhfe_cfg=dict(model_cfg.get("ag_lhfe", {}) or {}),
            verbose=verbose,
        )

'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--restore', action='store_true')
    args = p.parse_args()
    if not FACTORY.is_file():
        raise SystemExit(f'Not in the YOLO-tcy_v5 project root: missing {FACTORY}')
    if not (PROJECT / 'models/rgbt_rlsfa_gtga_centered_bhlr_model.py').is_file():
        raise SystemExit('Project is missing the expected GMLS/DMLS detector model')
    if not (PROJECT / 'ag_lhfe/model.py').is_file():
        raise SystemExit('Copy the ag_lhfe directory into the project root first')
    if args.restore:
        if not BACKUP.exists():
            raise SystemExit(f'No backup found: {BACKUP}')
        shutil.copy2(BACKUP, FACTORY)
        print('RESTORED', FACTORY)
        return
    src = FACTORY.read_text(encoding='utf-8')
    if TAG in src:
        ast.parse(src)
        print('Already installed; syntax valid:', FACTORY)
        return
    if src.count(MARK) != 1:
        raise SystemExit('Unsupported factory revision. No files changed; expected one exact variant anchor.')
    new_src = src.replace(MARK, BLOCK + MARK, 1)
    ast.parse(new_src)
    if args.check:
        print('[CHECK PASS] factory anchor, import files, syntax OK; no changes made')
        return
    if not BACKUP.exists():
        shutil.copy2(FACTORY, BACKUP)
    FACTORY.write_text(new_src, encoding='utf-8')
    print('[INSTALLED] new variant rlsfa_gtga_centered_ag_lhfe_v1')
    print('[BACKUP]', BACKUP)


if __name__ == '__main__':
    main()
