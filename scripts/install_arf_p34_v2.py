#!/usr/bin/env python3
"""Register a DISTINCT P3/P4 ARF variant in YOLO-tcy_v5's existing factory.

Requires the previous ARF-v1 installer already applied. Safe to rerun.
"""
from pathlib import Path
import ast
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
FACTORY = ROOT / 'models/gtga_bhlr_factory.py'
VARIANT_V1 = 'rlsfa_gtga_centered_reliability_fusion_v1'
VARIANT_P34 = 'rlsfa_gtga_centered_reliability_fusion_p34_v2'
MARKER = 'ARF_P34_FACTORY_V2'


def apply(code: str) -> str:
    if MARKER in code:
        return code
    # Existing v1 installer adds the old variant to the centered BHLR branch.
    old = '"' + VARIANT_V1 + '",'
    if code.count(old) != 1:
        raise RuntimeError('Existing ARF-v1 factory registration not found exactly once. '
                           'First run: python scripts/install_reliability_fusion.py')
    pos = code.index(old)
    code = code[:pos + len(old)] + ('\n        "' + VARIANT_P34 + '",') + code[pos + len(old):]

    old_clause = '        if variant == "' + VARIANT_V1 + '":'
    if code.count(old_clause) != 1:
        raise RuntimeError('Existing v1 model-class selector not found exactly once')
    replacement = """        # ARF_P34_FACTORY_V2: separate class and checkpoint variant.
        if variant == "rlsfa_gtga_centered_reliability_fusion_p34_v2":
            from models.rgbt_reliability_fusion_p34_model import (
                RGBTAlignmentReliabilityP34FusionDetectionModel,
            )
            model_class = RGBTAlignmentReliabilityP34FusionDetectionModel
            model_extra = {
                "reliability_fusion_cfg": dict(model_cfg.get("reliability_fusion", {}) or {})
            }
        elif variant == "rlsfa_gtga_centered_reliability_fusion_v1":"""
    code = code.replace(old_clause, replacement, 1)
    ast.parse(code)
    return code


def main():
    if not FACTORY.exists():
        raise FileNotFoundError(f'Run from YOLO-tcy_v5 project root; not found: {FACTORY}')
    before = FACTORY.read_text(encoding='utf-8')
    after = apply(before)
    if before == after:
        print('[OK] P34 variant already registered, unchanged')
        return
    bak = FACTORY.with_name('gtga_bhlr_factory.py.before_arf_p34_v2.bak')
    if not bak.exists():
        shutil.copy2(FACTORY, bak)
    FACTORY.write_text(after, encoding='utf-8')
    print('[OK] New variant registered:', VARIANT_P34)
    print('[OK] Old ARF-v1 and GMLS ADD variants left intact')
    print('[OK] Factory backup:', bak)

if __name__ == '__main__':
    main()
