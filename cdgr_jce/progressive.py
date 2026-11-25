import torch


def progressive_features(det,rgb,tir):
    from models.rgbt_model import get_layer_input
    return run_features(det,rgb,tir,get_layer_input)


def run_features(det,rgb,tir,get_layer_input):
    if rgb.shape!=tir.shape:raise ValueError('JCE requires same-grid RGB/TIR ROI')
    mode=det.jce_intervention
    if mode in ('rgb_only','zero_tir'):tir=torch.zeros_like(tir)
    if mode in ('tir_only','zero_rgb'):rgb=torch.zeros_like(rgb)
    rr=[];tt=[];fs={};r=rgb;t=tir
    for i,(rl,tl) in enumerate(zip(list(det.model)[:det.backbone_len],det.tir_backbone)):
        r=rl(get_layer_input(rl,r,rr));t=tl(get_layer_input(tl,t,tt))
        if i==4 and det.jce_weak_rgb:
            # Auxiliary pass only. No inverse keep-probability scaling.
            mask=(torch.rand(r.shape[:2]+(1,1),device=r.device)>=.5).to(r.dtype)
            scale=torch.empty((r.shape[0],1,1,1),device=r.device,dtype=r.dtype).uniform_(.15,.5)
            r=r*mask*scale
        if i in det.fusion_indices:
            j=det.jce_blocks[str(i)];j.intervention=mode
            # Explicit feature-level removal at EACH fusion point; zero-image alone leaves biases.
            if mode in ('rgb_only','zero_tir'):t=torch.zeros_like(t)
            if mode in ('tir_only','zero_rgb'):r=torch.zeros_like(r)
            z=j(r,t);fs[i]=z
            if str(i) in det.jce_feedback and mode not in ('add','no_feedback','rgb_only','tir_only','zero_rgb','zero_tir'):
                r,t=det.jce_feedback[str(i)](r,t,z)
        rr.append(r);tt.append(t)
    fused=list(rr)
    for i,z in fs.items():fused[i]=z
    return {'rgb_backbone':rr,'tir_backbone':tt,'fused_backbone':fused,'fusion_features':fs}
