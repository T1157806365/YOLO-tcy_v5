#!/usr/bin/env python3
"""Static preflight: verify independent 100e training configs and spatial patch.

Does not instantiate YOLO or change any files. No GPU required.
"""
from __future__ import annotations
import copy
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/'configs/experiments/v5_geohr_spatial_retrain'
MODES = ('control','spatial','spatial_guard')

def main() -> None:
    configs = {}
    for mode in MODES:
        path=FOLDER/f'uavcb_hvad_{mode}_officialPT_100e.yaml'
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        for key in ('init_weights','resume'):
            assert cfg['train'].get(key) is None, (mode,key,cfg['train'].get(key))
        assert cfg['model']['pretrained'] is True
        assert cfg['model']['fusion']=='add'
        assert cfg['model']['name']=='yolo26m-p2'
        assert cfg['train']['epochs']==100
        assert cfg['train']['batch']==24
        assert cfg['train']['batch_mode']=='global'
        assert cfg['train']['stage2']['enabled'] is False
        h=cfg['model']['hvad']
        assert h['enabled']
        assert h['spatial_hr_roi']==(mode!='control')
        assert h['confidence_gate_enabled']==(mode=='spatial_guard')
        configs[mode]=cfg
        print(f'[PASS] {mode}: official YOLO PT only, 100e, original GTGA/BHLR/TriGate/HVAD')
    # Other than the explicitly named module switches and output names,
    # both experiments must have exactly the same data/training settings.
    a=copy.deepcopy(configs['control'])
    b=copy.deepcopy(configs['spatial'])
    c=copy.deepcopy(configs['spatial_guard'])
    for x in (a,b,c):
        x['model']['hvad']['spatial_hr_roi']=None
        x['model']['hvad']['confidence_gate_enabled']=None
        x['output']=None
    assert a==b==c, 'Unexpected difference between experiment configurations'
    module=(ROOT/'hvad/module.py').read_text(encoding='utf-8')
    runtime=(ROOT/'hvad/runtime.py').read_text(encoding='utf-8')
    assert 'V5-GEOHR-SPATIAL-V1' in module
    assert 'native_roi' in module and 'hr_roi_mix' in module
    assert 'V5-GEOHR-SPATIAL-V1' in runtime
    pretrained=ROOT/'configs/weights/yolo26/yolo26m.pt'
    if not pretrained.exists() or pretrained.stat().st_size < 1024*1024:
        raise FileNotFoundError(f'Official model weights missing or invalid: {pretrained}')
    print(f'[PASS] Official weight exists: {pretrained} ({pretrained.stat().st_size/1048576:.1f} MB)')
    print('[PASS] All config and spatial patch static checks passed. CUDA training not tested here.')

if __name__=='__main__':
    main()
