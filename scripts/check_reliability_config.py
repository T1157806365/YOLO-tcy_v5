#!/usr/bin/env python3
"""Lightweight config/factory registration check. NO model build, GPU or dataset required."""
from pathlib import Path
import ast
import sys
import yaml

root=Path(__file__).resolve().parents[1]
p=root/'configs/experiments/reliability_fusion/uavcb_gmls_v2_reliability_yolo26m_1920_640_ddp2_bs32.yaml'
cfg=yaml.safe_load(p.read_text(encoding='utf-8'))
assert cfg['model']['variant']=='rlsfa_gtga_centered_reliability_fusion_v1'
assert cfg['model']['fusion']=='add'
assert cfg['model']['reliability_fusion']['enabled'] is True
assert cfg['model']['pretrained'] is True
assert cfg['model']['bhlr']['enabled'] is True
assert cfg['model']['rlsfa_gtga_centered']['coarse_gmls_enabled'] is True
assert cfg['model']['rlsfa_gtga_centered']['fine_translation_enabled'] is True
assert cfg['train']['batch']==32 and cfg['train']['batch_mode']=='global'
assert cfg['train']['epochs']==300 and cfg['train']['stage2']['epochs']==200
assert 'ahr2net' not in str(cfg).lower() and 'c4_' not in str(cfg).lower()
f=root/'models/gtga_bhlr_factory.py'
text=f.read_text(encoding='utf-8')
assert 'RL_FUSION_INTEGRATED_V1' in text, 'Run scripts/install_reliability_fusion.py first'
ast.parse(text,filename=str(f))
print('[OK] model:',cfg['model']['name'], 'variant:',cfg['model']['variant'])
print('[OK] device:',cfg['train']['device'],'DDP; global batch:',cfg['train']['batch'],'batch_mode:',cfg['train']['batch_mode'])
print('[OK] stage1:',cfg['train']['epochs'],'stage2:',cfg['train']['stage2']['epochs'])
print('[OK] GMLS-v2 + Fine + Scale + BHLR-v2 unchanged; ARF enabled; NO AHR2Net')
print('[OK] factory patch registered; run normal trainer')
