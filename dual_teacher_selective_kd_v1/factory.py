"""Independent factory: no mutation of YOLO-tcy_v5 files on disk."""
from __future__ import annotations
from pathlib import Path
from .model import DualTeacherSelectiveKD

VARIANT='dual_teacher_selective_kd_v1'

def validate_cfg(cfg,*,require_teachers=True):
    m=cfg.get('model',{});t=cfg.get('train',{});d=cfg.get('data',{});
    teach=cfg.get('dual_teachers',{}); kd=cfg.get('selective_kd',{})
    if m.get('variant')!=VARIANT:raise ValueError(f'model.variant must be {VARIANT}')
    if m.get('name')!='yolo26m' or int(m.get('nc',0))!=1:raise ValueError('This release supports YOLO26m, nc=1')
    if not isinstance(m.get('pretrained'),bool):raise ValueError('model.pretrained must be a YAML boolean')
    if m.get('fusion')!='add':raise ValueError('Use compatibility fusion: add (actual fusion is new P3 residual)')
    if list(m.get('fusion_indices',[]))!=[4,6,10]:raise ValueError('Expected P3/P4/P5 YOLO26m fusion indices [4,6,10]')
    if not m.get('rlsfa_gtga_centered',{}).get('enabled',False):raise ValueError('Alignment must be enabled')
    ac=m['rlsfa_gtga_centered']
    if not ac.get('coarse_local_soft_enabled',False) or ac.get('coarse_gmls_enabled',False) or ac.get('gmls_v2_correction_enabled',False):
        raise ValueError('This is the DMLS+Fine+Scale student, NOT a GMLS-v2 student')
    if int(d['rgb_semantic_imgsz'])!=int(d['tir_imgsz']):raise ValueError('RGB semantic and TIR must match')
    if m.get('bhlr',{}).get('enabled',False):
        if int(d['rgb_imgsz'])!=int(m['bhlr'].get('high_to_semantic_ratio',3))*int(d['rgb_semantic_imgsz']):
            raise ValueError('BHLR RGB high/semantic ratio mismatch')
    if int(t.get('patience',0))<=0 or int(t.get('stage2',{}).get('patience',0))<=0:
        raise ValueError('Both stage1 and stage2 patience must be positive (e.g. 50)')
    if t.get('resume') is not None:raise ValueError('Clean trainer has no resume; use resume: null')
    if not kd.get('enabled',False):raise ValueError('selective_kd.enabled must be true')
    if require_teachers:
        for k in ('rgb_weights','tir_weights'):
            p=Path(teach.get(k,'')).expanduser()
            if not p.is_file():raise FileNotFoundError(f'dual_teachers.{k} not found: {p}')
        if Path(teach['rgb_weights']).resolve()==Path(teach['tir_weights']).resolve():
            raise ValueError('RGB/TIR teacher weights cannot be the same file')
    for k in ('rgb','tir'):
        p=Path(d.get(k,''));
        if not p.is_file():raise FileNotFoundError(f'Dataset YAML {k} missing: {p}')
    if m['pretrained']:
        from utils.config import resolve_model
        w=Path(resolve_model(m['name'],pretrained=True)['weights'])
        if not w.is_file():raise FileNotFoundError(f'Official student pretrained weights missing: {w}')
    if int(teach.get('forward_microbatch',2))<1:raise ValueError('forward_microbatch >= 1')
    return cfg


def build_model_from_config(cfg,pretrained=None,fusion_override=None,names=None,nc=None,verbose=True,
                            attach_teachers=True):
    m=cfg['model'];d=cfg['data'];b=m['bhlr'];loss=cfg.get('loss',{})
    if pretrained is None:pretrained=m['pretrained']
    if names is None:names={0:'UAV'}
    if nc is None:nc=m['nc']
    model=DualTeacherSelectiveKD(
        model_name=m['name'],nc=int(nc),pretrained=bool(pretrained),
        fusion=str(fusion_override or m['fusion']),fusion_indices=m['fusion_indices'],
        names=names,semantic_imgsz=int(d['rgb_semantic_imgsz']),
        align_mode=str(m.get('align_mode','bilinear')),
        rlsfa_gtga_centered_cfg=dict(m['rlsfa_gtga_centered']),
        rlsfa_gtga_centered_loss_cfg=dict(loss.get('alignment',{})),
        bhlr_enabled=bool(b.get('enabled',True)),bhlr_scales=b.get('scales',[3]),
        high_to_semantic_ratio=int(b.get('high_to_semantic_ratio',3)),
        detail_channels=int(b.get('detail_channels',48)),
        stem_channels=int(b.get('stem_channels',16)),
        guide_channels=int(b.get('guide_channels',24)),
        bhlr_guidance_mode=str(b.get('guidance_mode','alignment')),
        bhlr_detach_alignment_guidance=bool(b.get('detach_alignment_guidance',True)),
        selective_cfg=dict(cfg['selective_kd']),teacher_cfg=dict(cfg['dual_teachers']),
        verbose=bool(verbose))
    # Teachers are deliberately lazy; forward train batch builds frozen teachers.
    # attach_teachers=False is used by val; NO teacher checkpoint needed at test.
    if not attach_teachers:
        model.teacher_cfg={}
    return model
