"""Standalone model factory; does not patch the existing model registry on disk."""
from __future__ import annotations
from pathlib import Path
from .model import OnlineRGBTKD


def validate_cfg(cfg):
    m = cfg.get('model', {})
    d = cfg.get('data', {})
    t = cfg.get('train', {})
    kd = cfg.get('online_kd', {})
    assert m.get('variant') == 'online_rgbt_kd_v1', 'Wrong model.variant'
    assert m.get('name') == 'yolo26m', 'This tested config is specifically YOLO26m'
    assert int(m.get('nc', 0)) == 1, 'Expected single-class UAV detection'
    assert isinstance(m.get('pretrained'), bool), 'model.pretrained must be YAML boolean true/false'
    assert m.get('fusion') == 'add', 'Parent wrapper must initialize ADD compatibility'
    assert m.get('fusion_indices', [4,6,10]) == [4,6,10], 'Expected YOLO26m P3/P4/P5 [4,6,10]'
    assert (not m.get('bhlr', {}).get('enabled', True)) or (int(d.get('rgb_imgsz', 0)) == 3 * int(d.get('rgb_semantic_imgsz', -1))), 'BHLR 3x requires rgb_imgsz=3*semantic'
    assert int(d.get('rgb_semantic_imgsz', 0)) == int(d.get('tir_imgsz', -1)), 'This implementation expects semantic RGB/TIR same input side'
    assert m.get('rlsfa_gtga_centered', {}).get('enabled') is True
    assert m.get('rlsfa_gtga_centered', {}).get('coarse_gmls_enabled') is False
    assert m.get('rlsfa_gtga_centered', {}).get('gmls_v2_correction_enabled') is False
    assert m.get('rlsfa_gtga_centered', {}).get('coarse_local_soft_enabled') is True
    assert int(t.get('patience', 0)) >= 0
    assert int(t.get('stage2', {}).get('patience', 0)) >= 0
    assert kd.get('enabled', True) is True, 'This is the online distillation configuration'
    assert t.get('resume') is None, 'Base trainer does not implement resume'
    for k in ('rgb', 'tir'):
        p = Path(d[k]).expanduser()
        if not p.exists():
            raise FileNotFoundError(f'Dataset config not found: {p}')
    if m['pretrained']:
        from utils.config import resolve_model
        w = Path(resolve_model(m['name'], pretrained=True)['weights'])
        if not w.is_file():
            raise FileNotFoundError(f'Official YOLO26m pretraining weight missing: {w}. No auto-download; supply locally or set pretrained: false')
    return cfg


def build_model_from_config(cfg, pretrained=None, fusion_override=None, names=None, nc=None, verbose=True):
    m = cfg['model']
    d = cfg['data']
    lcfg = cfg.get('loss', {})
    bhlr = m['bhlr']
    if pretrained is None:
        pretrained = m['pretrained']
    if nc is None:
        nc = m.get('nc', 1)
    if names is None:
        names = {0:'UAV'}
    return OnlineRGBTKD(
        model_name=str(m['name']), nc=int(nc), pretrained=bool(pretrained),
        fusion=str(fusion_override or m['fusion']),
        fusion_indices=m.get('fusion_indices', [4,6,10]),
        align_mode=m.get('align_mode', 'bilinear'), names=names,
        semantic_imgsz=int(d['rgb_semantic_imgsz']),
        rlsfa_gtga_centered_cfg=dict(m['rlsfa_gtga_centered']),
        rlsfa_gtga_centered_loss_cfg=dict(lcfg.get('alignment', {})),
        bhlr_enabled=bool(bhlr.get('enabled', True)),
        bhlr_scales=bhlr.get('scales', [3]),
        high_to_semantic_ratio=int(bhlr.get('high_to_semantic_ratio', 3)),
        detail_channels=int(bhlr.get('detail_channels', 48)),
        stem_channels=int(bhlr.get('stem_channels', 16)),
        guide_channels=int(bhlr.get('guide_channels', 24)),
        bhlr_guidance_mode=str(bhlr.get('guidance_mode', 'alignment')),
        bhlr_detach_alignment_guidance=bool(bhlr.get('detach_alignment_guidance', True)),
        kd_cfg=dict(cfg.get('online_kd', {})),
        verbose=bool(verbose),
    )
