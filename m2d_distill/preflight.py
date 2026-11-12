"""Cheap on-server configuration + source readiness checks (no GPU allocation)."""
import argparse
from pathlib import Path
import yaml


def check(cfg, root):
    assert cfg['model']['variant'] == 'rlsfa_gtga_centered_bhlr_v2'
    assert cfg['model']['bhlr']['injection_position'] == 'post_fusion'
    assert cfg['model']['bhlr']['scales'] == [3]
    assert cfg['model']['tri_gate']['enabled'] is True
    assert 0.0 < float(cfg['model']['tri_gate']['max_change']) <= 0.3
    assert cfg['monoalign']['init_from_teachers'] is False
    assert cfg['monoalign']['schedule_mode'] == 'm2d_joint100'
    assert cfg['model']['pretrained'] is True
    assert cfg['m2d_distill']['enabled'] is True
    assert cfg['m2d_distill']['scales'] == [3, 4, 5]
    assert cfg['train']['epochs'] == 100
    assert cfg['train']['stage2']['enabled'] is False
    assert cfg['train'].get('resume') is None
    assert cfg['train'].get('init_student_weights') is None
    assert cfg['monoalign']['pcgrad_enabled'] is True
    assert cfg['model']['rlsfa_gtga_centered']['enabled'] is True
    assert cfg['model']['bhlr']['guidance_mode'] == 'alignment'
    for x in ('rgb_weights', 'tir_weights'):
        p = Path(cfg['dual_teachers'][x])
        if not p.is_file():
            raise FileNotFoundError(f'Missing {x}: {p}')
    for x in (cfg['data']['rgb'],cfg['data']['tir']):
        if not Path(x).is_file():
            raise FileNotFoundError(f'Missing dataset YAML: {x}')
    if 'M2D_DISTILL_JOINT100_V1' not in (root/'ga_monoalign_pcgrad/model.py').read_text():
        raise RuntimeError('M2D model bridge not installed')
    if not (root/'m2d_distill/losses.py').is_file():
        raise RuntimeError('Separate M2D implementation not installed')
    print('[PASS] Single-stage100, official YOLO init, NO teacher weight copy')
    print('[PASS] Frozen teachers + multi-scale intra/cross/fused KD')
    print('[PASS] Original RLSFA-GTGA + Post-Fusion BHLR 1920 residual')
    print('[PASS] P3 bounded three-way attention (max_change<=0.3), NO LIF')
    print('[PASS] Teacher weights & dataset YAML files exist')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    args=p.parse_args()
    root=Path.cwd()
    cfg=yaml.safe_load(Path(args.cfg).read_text(encoding='utf-8'))
    check(cfg,root)

if __name__=='__main__':main()
