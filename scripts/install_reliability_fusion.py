#!/usr/bin/env python3
"""Idempotently register the new model variant in the EXISTING model factory.

Run in the repository root after extracting the ZIP.  The old centered
model branch remains compatible; other branches stay unchanged.
"""
from pathlib import Path
import ast
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
FACTORY=ROOT/'models/gtga_bhlr_factory.py'
MARKER='RL_FUSION_INTEGRATED_V1'
VARIANT='rlsfa_gtga_centered_reliability_fusion_v1'

def apply(text):
    if MARKER in text:
        return text
    # Only change the selected historical centered branch.
    pat=(r'(?P<head>\b(?:if|elif)\s+variant\s+in\s*\{\s*'
         r'[\"\']rlsfa_gtga_centered_bhlr_v2[\"\']\s*,\s*'
         r'[\"\']rlsfa_gtga_centered[\"\']\s*,?\s*\}\s*:)')
    found=list(re.finditer(pat,text))
    if len(found)!=1:
        raise RuntimeError(f'Expected exactly one centered variant branch, found {len(found)}. '
                           'Factory may differ from supported v5; no files changed.')
    m=found[0]
    h=m.group('head')
    nh=h.replace('"rlsfa_gtga_centered",','"rlsfa_gtga_centered",\n        "'+VARIANT+'",')
    if nh==h: raise RuntimeError('Could not insert variant identifier')
    text=text[:m.start()]+nh+text[m.end():]

    from_idx=text.index(nh)
    original='        model = RGBTRLSFAGTGACenteredBHLRV2DetectionModel('
    n=text.count(original,from_idx)
    if n<1: raise RuntimeError('Could not find centered model constructor')
    call_idx=text.index(original,from_idx)
    # Additional guard against wrong target before legacy branch.
    legacy=text.find('# 2. Legacy RLSFA-GTGA',from_idx)
    if legacy!=-1 and call_idx>legacy:raise RuntimeError('Constructor not in centered branch')
    constructor='''        # RL_FUSION_INTEGRATED_V1: choose model subclass ONLY for the new YAML.
        if variant == "'''+VARIANT+'''":
            from models.rgbt_reliability_fusion_model import (
                RGBTAlignmentReliabilityFusionDetectionModel,
            )
            model_class = RGBTAlignmentReliabilityFusionDetectionModel
            model_extra = {
                "reliability_fusion_cfg": dict(model_cfg.get("reliability_fusion", {}) or {})
            }
        else:
            model_class = RGBTRLSFAGTGACenteredBHLRV2DetectionModel
            model_extra = {}

        model = model_class('''
    text=text[:call_idx]+text[call_idx:].replace(original,constructor,1)

    call_idx=text.index('        model = model_class(',from_idx)
    # Find the matching outer closing parenthesis based on indentation.
    close_idx=text.find('\n        )',call_idx)
    if close_idx==-1:raise RuntimeError('Centered constructor closing parenthesis not found')
    fragment=text[call_idx:close_idx]
    pos=fragment.rfind('            verbose=verbose,')
    if pos<0:raise RuntimeError('Cannot find verbose=verbose in centered constructor')
    fragment=fragment[:pos]+fragment[pos:].replace('            verbose=verbose,',
                     '            verbose=verbose,\n            **model_extra,',1)
    text=text[:call_idx]+fragment+text[close_idx:]
    ast.parse(text,filename=str(FACTORY))
    return text


def main():
    if not FACTORY.is_file():
        raise FileNotFoundError('Run this script INSIDE YOLO-tcy_v5 project root: '+str(FACTORY))
    old=FACTORY.read_text(encoding='utf-8')
    updated=apply(old)
    if old==updated:
        print('[OK] Factory already registered; no changes needed.')
        return
    bak=FACTORY.with_name('gtga_bhlr_factory.py.before_reliability_fusion.bak')
    if not bak.exists():shutil.copy2(FACTORY,bak)
    FACTORY.write_text(updated,encoding='utf-8')
    print('[OK] Factory patched:',FACTORY)
    print('[OK] Original backup:',bak)
    print('[OK] Previous training and validation scripts preserved unchanged')

if __name__=='__main__':main()
