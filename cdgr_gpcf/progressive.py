import torch

def progressive_features(det,rgb,tir):
    from models.rgbt_model import get_layer_input
    return run_progressive(det,rgb,tir,get_layer_input)

def run_progressive(det,rgb,tir,get_layer_input):
    # Native YOLO `from` indexing preserved, including non-sequential connections.
    if rgb.shape!=tir.shape:raise ValueError('Expected same RGB/TIR ROI shape')
    if det.gpcf_intervention=='rgb_only':tir=torch.zeros_like(tir)
    if det.gpcf_intervention=='tir_only':rgb=torch.zeros_like(rgb)
    rr=[];tt=[];r=rgb;t=tir;guides=det._gpcf_guides
    for i,(rl,tl) in enumerate(zip(list(det.model)[:det.backbone_len],det.tir_backbone)):
        r=rl(get_layer_input(rl,r,rr));t=tl(get_layer_input(tl,t,tt))
        if str(i) in det.gpcf_blocks:
            block=det.gpcf_blocks[str(i)];g=guides.get(i)
            # Avoid any mistaken pairing of geometry to a filtered Anti-UAV batch.
            if g is not None and g.shape[0]!=len(r):
                ids=det._gpcf_keep
                if ids is None or int(ids.numel())!=len(r):raise RuntimeError('Unexpected ROI subset; refusing wrong geometry mapping')
                g=g.index_select(0,ids)
            if det.gpcf_intervention in ('rgb_only','tir_only'):
                # For these diagnostics no cross-transfer of zero-input biases; bypass both GPCF blocks.
                pass
            else:r,t=block(r,t,g)
        rr.append(r);tt.append(t)
    if det.gpcf_intervention=='rgb_only':
        fused=list(rr);fs={i:rr[i] for i in det.fusion_indices}
    elif det.gpcf_intervention=='tir_only':
        fused=list(rr);fs={i:tt[i] for i in det.fusion_indices}
        for i in det.fusion_indices:fused[i]=tt[i]
    else:fused,fs=det.fuse_backbone_features(rr,tt)
    return {'rgb_backbone':rr,'tir_backbone':tt,'fused_backbone':fused,'fusion_features':fs}

