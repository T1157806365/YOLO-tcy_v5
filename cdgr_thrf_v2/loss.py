"""Training-only actual TIR GT -> inverse CDGR -> local RGB ROI coordinates.
The geometry used to form supervision is detached. TIR-only GT is retained;
RGB-positive/TIR-empty frames are ignored conservatively. Empty paired frames
are background; partially visible/unusable TIR GT makes the ROI ignored.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F

@torch.no_grad()
def local_thermal_targets(batch,ex,min_visible=.6,min_valid=.7):
    roi=ex['roi'];info=ex['alignment_info'];B=batch['rgb_img'].shape[0];K=int(ex['rois_per_image'])
    ref=roi.xy0.detach().float();dev=ref.device
    s=info['sampling_scale'].detach().float().reshape(B,2).clamp_min(1e-6)
    t=info['sampling_translation'].detach().float().reshape(B,2)
    ht,wt=batch['tir_img'].shape[-2:];hs,ws=batch['rgb_semantic_img'].shape[-2:]
    boxes=batch['tir_bboxes'].detach().float().to(dev);idx=batch['tir_batch_idx'].long().to(dev)
    rgb_idx=batch['rgb_batch_idx'].long().to(dev)
    valid=(roi.tir_valid.detach().reshape(-1)>=min_valid).clone()
    result=[];indices=[]
    def meta(m):
        r=m['ratio'];r=float(r[0] if isinstance(r,(list,tuple)) else r)
        return r,ref.new_tensor(m['pad'])
    for i in range(B):
        bb=boxes[idx==i]
        if len(bb)==0:
            if bool((rgb_idx==i).any()):valid[i*K:(i+1)*K]=False
            continue
        # YOLO xywh is normalized by W/H; grid_sample uses W-1/H-1.
        scale=ref.new_tensor([wt,ht])/ref.new_tensor([max(wt-1,1),max(ht-1,1)])
        qlo=(bb[:,:2]-bb[:,2:]/2)*scale;qhi=(bb[:,:2]+bb[:,2:]/2)*scale
        semscale=ref.new_tensor([max(ws-1,1),max(hs-1,1)])
        plo=(qlo-t[i])/s[i]*semscale;phi=(qhi-t[i])/s[i]*semscale
        rh,ph=meta(batch['rgb_ratio_pad'][i]);rs,ps=meta(batch['rgb_semantic_ratio_pad'][i])
        lo=(plo-ps)/rs*rh+ph;hi=(phi-ps)/rs*rh+ph
        for j in range(K):
            n=i*K+j
            if not bool(valid[n]):continue
            ll=(lo-ref[n]).clamp(0,roi.size);uu=(hi-ref[n]).clamp(0,roi.size)
            wh=(uu-ll).clamp_min(0)
            frac=wh.prod(-1)/(hi-lo).clamp_min(1e-6).prod(-1)
            center=(lo+hi)/2-ref[n]
            keep=(frac>=min_visible)&(wh.min(1).values>=1)&((center>=0)&(center<roi.size)).all(1)
            if bool(((frac>0)&~keep).any()):
                valid[n]=False;continue
            if bool(keep.any()):
                result.append(torch.cat((center[keep],wh[keep]),1)/roi.size)
                indices.append(torch.full((int(keep.sum()),),n,device=dev,dtype=torch.long))
    return (torch.cat(result) if result else ref.new_zeros((0,4)),
            torch.cat(indices) if indices else torch.empty(0,dtype=torch.long,device=dev),valid)


def heat_loss(logits,boxes,indices,valid):
    n,_,h,w=logits.shape;target=logits.new_zeros((n,h,w),dtype=torch.float32)
    ys=torch.arange(h,device=logits.device,dtype=torch.float32)[:,None]
    xs=torch.arange(w,device=logits.device,dtype=torch.float32)[None,:]
    for i,box in zip(indices.tolist(),boxes):
        cx,cy,bw,bh=box
        sigma=(torch.sqrt((bw*w*bh*h).clamp_min(1e-6))*.24).clamp(1.,4.)
        # P3 feature centers approximate stride-8 cell centers.
        g=torch.exp(-.5*((xs-(cx*w-.5)).square()+(ys-(cy*h-.5)).square())/sigma.square())
        target[i]=torch.maximum(target[i],g)
    y=target[:,None];z=logits.float();p=z.sigmoid();m=valid[:,None,None,None].float()
    pos=-(m*y*(1-p).square()*F.logsigmoid(z)).sum()/(m*y).sum().clamp_min(1.)
    neg=-(m*(1-y).pow(4)*p.square()*F.logsigmoid(-z)).sum()/(m.sum()*h*w).clamp_min(1.)
    return pos+.25*neg,{'tir_supervised_roi_frac':valid.float().mean().detach(),
                         'tir_positive_rois':(y.flatten(1).amax(1)>0).float().sum().detach()}
