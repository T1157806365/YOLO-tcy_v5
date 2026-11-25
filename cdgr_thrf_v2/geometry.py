"""Per-ROI CDGR geometry descriptors, from predictions only (no GT leakage)."""
from __future__ import annotations
import torch


def roi_cdgr_descriptor(batch,ex,*,center_norm=24.0,log_norm=0.5,detach=True):
    info=ex['alignment_info']
    need=('sampling_scale','sampling_translation','sampling_scale_raw_gtga',
          'sampling_translation_raw_gtga','coarse_anchor')
    missing=[k for k in need if k not in info]
    if missing:raise KeyError(f'CDGR geometry missing {missing}')
    s=info['sampling_scale'].float().reshape(-1,2)
    t=info['sampling_translation'].float().reshape(-1,2)
    s0=info['sampling_scale_raw_gtga'].float().reshape(-1,2)
    t0=info['sampling_translation_raw_gtga'].float().reshape(-1,2)
    a=info['coarse_anchor'].float().reshape(-1,2)
    b=batch['rgb_img'].shape[0];k=int(ex['rois_per_image'])
    if len(s)!=b or k!=2:raise ValueError(f'Expected B x Top2 geometry, B={b} K={k}')
    hs,ws=batch['rgb_semantic_img'].shape[-2:]
    hh,wh=batch['rgb_img'].shape[-2:]
    ht,wt=batch['tir_img'].shape[-2:]
    delta=((s*a+t)-(s0*a+t0))*s.new_tensor([ws-1,hs-1])/max(center_norm,1e-6)
    log=(s.clamp_min(1e-6)/s0.clamp_min(1e-6)).log()/max(log_norm,1e-6)
    density=(s*s.new_tensor([wt/max(wh,1),ht/max(hh,1)])).clamp_min(1e-6).log()
    shared=torch.cat((delta,log,density),dim=1).repeat_interleave(k,dim=0)
    valid=ex['roi'].tir_valid.float().reshape(-1,1).detach()
    ctx=torch.cat((shared,valid),dim=1)
    if ctx.shape!=(b*k,7):raise RuntimeError(f'Geometry shape {ctx.shape} != {(b*k,7)}')
    return ctx.detach() if detach else ctx
