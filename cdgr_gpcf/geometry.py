"""Project detached CDGR support and valid-source masks into each RGB ROI grid."""
import torch
import torch.nn.functional as F

def _meta(metas, ref):
    ratios=[];pads=[]
    for m in metas:
        r=m['ratio']
        if isinstance(r,(tuple,list)):
            if len(r)!=2 or abs(float(r[0])-float(r[1]))>1e-5:raise ValueError('Expected isotropic letterbox')
            r=r[0]
        if float(r)<=0:raise ValueError('Invalid letterbox ratio')
        ratios.append(float(r));pads.append(m['pad'])
    return ref.new_tensor(ratios).view(-1,1,1,1),ref.new_tensor(pads).view(-1,1,1,2)

@torch.no_grad()
def roi_guidance(batch,ex,hw):
    info=ex['alignment_info']; support=info['multi_target_guidance'].detach().float()
    b=len(batch['rgb_img']);k=int(ex['rois_per_image']);n=b*k
    origin=ex['roi'].xy0.detach().float();size=int(ex['roi'].size)
    if origin.shape!=(n,2) or support.shape[:2]!=(b,3):raise ValueError('Unexpected ROI geometry/support')
    h,w=hw;sh,sw=batch['rgb_semantic_img'].shape[-2:]
    # Feature-cell centers on the corresponding ROI canvas; no extra warp of RGB features.
    y=(torch.arange(h,device=origin.device,dtype=torch.float32)+.5)*size/h-.5
    x=(torch.arange(w,device=origin.device,dtype=torch.float32)+.5)*size/w-.5
    yy,xx=torch.meshgrid(y,x,indexing='ij');xy=origin[:,None,None,:]+torch.stack((xx,yy),-1)[None]
    rh,ph=_meta([m for m in batch['rgb_ratio_pad'] for _ in range(k)],origin)
    rs,ps=_meta([m for m in batch['rgb_semantic_ratio_pad'] for _ in range(k)],origin)
    p=((xy-ph)/rh*rs+ps)/origin.new_tensor([max(sw-1,1),max(sh-1,1)])
    s=info['sampling_scale'].detach().float().reshape(b,2).repeat_interleave(k,0)
    t=info['sampling_translation'].detach().float().reshape(b,2).repeat_interleave(k,0)
    q=p*s[:,None,None,:]+t[:,None,None,:]
    valid=((q>=0)&(q<=1)).all(-1)&((p>=0)&(p<=1)).all(-1)
    # Validity is in-canvas validity, not a verified non-padding mask.
    crop=F.grid_sample(support.repeat_interleave(k,0),2*p-1,mode='bilinear',padding_mode='zeros',align_corners=True).clamp(0,1)
    return torch.cat((crop,valid[:,None].float()),1).detach()
