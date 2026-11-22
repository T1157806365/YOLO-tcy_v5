"""Teacher feature-response geometry losses for CGD-10 V2.

Native RGB-semantic/TIR teacher P3 responses are built by the teacher runtime.
All teacher responses/targets are detached. Differentiability is limited to
student router logits, coarse candidate offsets, and final GTGA sampling S,t.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _zero_like(*tensors):
    out = None
    for t in tensors:
        if torch.is_tensor(t):
            v = t.float().sum() * 0.0
            out = v if out is None else out + v
    if out is None:
        raise ValueError('At least one tensor required')
    return out


def response_moments(prob: torch.Tensor):
    """Return normalized center and per-axis standard deviations [B,2]."""
    if prob.ndim != 4 or prob.shape[1] != 1:
        raise ValueError(f'Expected [B,1,H,W] probability map, got {tuple(prob.shape)}')
    b,_,h,w=prob.shape
    p=prob.detach().float().clamp_min(0)
    p=p/(p.sum(dim=(2,3),keepdim=True).clamp_min(1e-12))
    yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=p.device),
                         torch.linspace(0,1,w,device=p.device), indexing='ij')
    x=(p[:,0]*xx).sum((1,2)); y=(p[:,0]*yy).sum((1,2))
    cx=x[:,None,None];cy=y[:,None,None]
    sx=torch.sqrt((p[:,0]*(xx[None]-cx).square()).sum((1,2)).clamp_min(1e-8))
    sy=torch.sqrt((p[:,0]*(yy[None]-cy).square()).sum((1,2)).clamp_min(1e-8))
    return torch.stack((x,y),1).detach(),torch.stack((sx,sy),1).detach()


def response_from_p3(feature: torch.Tensor, boxes: torch.Tensor, *,
                     roi_pad_cells=3., window_expand=1.25,
                     response_temperature=0.75):
    """Create *real P3 feature energy* spatial attention around paired native GT.

    boxes are normalized xywh in each teacher's own 640-by-640 canvas.
    The ROI MASK comes from GT, but spatial energy and moments come from P3,
    not synthetic box Gaussians. No gradients through teacher features.
    """
    with torch.no_grad():
        f=feature.detach().float()
        b,_,h,w=f.shape
        boxes=boxes.detach().float().reshape(b,4)
        yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=f.device),
                             torch.linspace(0,1,w,device=f.device),indexing='ij')
        xc=boxes[:,0,None,None]; yc=boxes[:,1,None,None]
        half_w=torch.maximum(boxes[:,2,None,None].clamp_min(1e-4)*float(window_expand)*0.5,
                             torch.full_like(xc,float(roi_pad_cells)/max(w-1,1)))
        half_h=torch.maximum(boxes[:,3,None,None].clamp_min(1e-4)*float(window_expand)*0.5,
                             torch.full_like(yc,float(roi_pad_cells)/max(h-1,1)))
        mask=(xx[None]-xc).abs()<=half_w
        mask &= (yy[None]-yc).abs()<=half_h
        # Mean channel energy retains local evidence, unlike a Gaussian GT box.
        energy=f.square().mean(1)
        nm=mask.sum((1,2)).clamp_min(1)
        mean=(energy*mask).sum((1,2))/nm
        var=((energy-mean[:,None,None]).square()*mask).sum((1,2))/nm
        std=var.clamp_min(1e-8).sqrt()
        scores=((energy-mean[:,None,None])/std[:,None,None]).clamp(-6,6)
        scores=scores/max(float(response_temperature),0.05)
        scores=scores.masked_fill(~mask,-1e4)
        p=torch.softmax(scores.flatten(1),dim=1).reshape(b,1,h,w)
        # Estimate concentration relative to uniform inside the target ROI.
        peak=(p.flatten(1).max(1).values * nm)
        concentration=((peak-1.)/4.).clamp(0,1)
        center,std_xy=response_moments(p)
        return {'heatmap':p.detach(),'center':center,'sigma':std_xy,
                'concentration':concentration.detach(),
                'roi_pixels':nm.detach()}


def _teacher_center_target(gt_box, teacher_center, max_residual_px=10., canvas_size=640.):
    """Do not let a spurious teacher attention peak overrule annotated geometry."""
    gt=gt_box.detach().float()[:,:2]
    delta=(teacher_center.detach().float()-gt)*float(canvas_size)
    # Limit the magnitude of the two-dimensional residual, not each axis.
    norm=torch.linalg.vector_norm(delta,dim=1,keepdim=True)
    delta=delta*torch.clamp(float(max_residual_px)/norm.clamp_min(1e-6),max=1.)
    return (gt+delta/float(canvas_size)).detach()


def route_offset_response_kd(route_logits,gmls_offset,dmls_offset,
                             rgb_gt,tir_gt,rgb_center,tir_center,quality,*,
                             feature_hw=(80,80),route_temperature=0.65,
                             route_gt_margin=1.5,route_teacher_strength=0.3,
                             sigma_features=4.0,offset_beta_features=2.,
                             max_teacher_residual_px=10.):
    """Route logits (ambiguous GT cases) + differentiable candidate offset KD.

    V1 detached the candidate offsets and only learned route logits. V2 makes
    the teacher-derived center displacement a direct gradient target for the
    best candidate, without altering the hard route used by GeoBox inference.
    """
    logits=route_logits.float().reshape(-1,2)
    b=logits.shape[0]
    offsets=torch.stack((gmls_offset.float().reshape(b,2),
                         dmls_offset.float().reshape(b,2)),dim=1)
    q=quality.detach().float().reshape(b).clamp(0,1)
    h,w=feature_hw
    pix=logits.new_tensor([max(w-1,1),max(h-1,1)])
    rgb_c=_teacher_center_target(rgb_gt,rgb_center,max_teacher_residual_px)
    tir_c=_teacher_center_target(tir_gt,tir_center,max_teacher_residual_px)
    delta_t=(tir_c-rgb_c)*pix
    delta_gt=(tir_gt.detach().float()[:,:2]-rgb_gt.detach().float()[:,:2])*pix
    sig=max(float(sigma_features),0.2)
    sg=-.5*((offsets.detach()-delta_gt[:,None,:])/sig).square().sum(2)
    st=-.5*((offsets.detach()-delta_t[:,None,:])/sig).square().sum(2)
    teacher_scores=sg+float(route_teacher_strength)*q[:,None]*st
    soft_teacher=F.softmax(teacher_scores/max(float(route_temperature),.05),dim=1).detach()
    ambiguous=((sg[:,0]-sg[:,1]).abs()<=float(route_gt_margin))&(q>0)
    eligible=ambiguous.float()
    route_per=F.kl_div(F.log_softmax(logits,dim=1),soft_teacher,reduction='none').sum(1)
    route_loss=(route_per*q*eligible).sum()/eligible.sum().clamp_min(1.)
    # Offsets themselves receive student gradients: teacher-weighted best path.
    off_err=F.smooth_l1_loss((offsets-delta_t[:,None,:])/max(float(offset_beta_features),0.1),
                            torch.zeros_like(offsets),beta=1.,reduction='none').mean(-1)
    per=(off_err*soft_teacher).sum(-1)
    offset_loss=(per*q).sum()/(q>0).float().sum().clamp_min(1.)
    return route_loss,offset_loss,{
        'route_eligible':eligible.mean().detach(),
        'route_agreement':((logits.argmax(1)==soft_teacher.argmax(1)).float()*eligible).sum().detach()/eligible.sum().clamp_min(1.),
        'route_target_dmls':(soft_teacher[:,1]*eligible).sum().detach()/eligible.sum().clamp_min(1.),
        'offset_error_features':((offsets.detach()-delta_t[:,None,:]).norm(dim=-1)*soft_teacher).sum(-1).mul(q).sum().detach()/q.sum().clamp_min(1e-6),
        'offset_valid_rate':(q>0).float().mean().detach(),
        'teacher_delta_center_px':((tir_c-rgb_c)-(tir_gt[:,:2]-rgb_gt[:,:2])).norm(dim=1).mul(640).mean().detach(),
    }


def warp_feature_response(tir_map,scale,trans):
    """q=S*p+t samples native TIR P3 heatmap into RGB-sem P3 grid."""
    b,_,h,w=tir_map.shape
    scale=scale.float().reshape(b,2)
    trans=trans.float().reshape(b,2)
    yy,xx=torch.meshgrid(torch.linspace(0,1,h,device=tir_map.device),
                         torch.linspace(0,1,w,device=tir_map.device),indexing='ij')
    base=torch.stack((xx,yy),dim=-1)[None].expand(b,-1,-1,-1)
    q=base*scale[:,None,None,:]+trans[:,None,None,:]
    return F.grid_sample(tir_map.detach().float(),q*2-1,mode='bilinear',
                         padding_mode='zeros',align_corners=True)


def final_feature_warp_kd(scale,trans,rgb_resp,tir_resp,rgb_gt,tir_gt,quality,*,
                          canvas_size=640.,max_teacher_residual_px=10.,
                          center_norm_px=16.,size_weight=.15,overlap_weight=.05):
    """Teacher P3 response moments supervise student final S,t, not synthetic boxes.

    Geometric consistency is measured in TIR coordinates: S*mu_rgb+t -> mu_tir.
    Soft overlap uses actual P3 spatial response maps (not Gaussian GT boxes).
    """
    s=scale.float().reshape(-1,2).clamp_min(1e-5)
    t=trans.float().reshape(-1,2)
    q=quality.detach().float().reshape(-1).clamp(0,1)
    rcenter=_teacher_center_target(rgb_gt,rgb_resp['center'],max_teacher_residual_px,canvas_size)
    tcenter=_teacher_center_target(tir_gt,tir_resp['center'],max_teacher_residual_px,canvas_size)
    # Student transform maps RGB coordinate p to TIR sampling position q.
    residual=(s*rcenter+t-tcenter)*float(canvas_size)
    center=F.smooth_l1_loss(residual/max(float(center_norm_px),.1),
                            torch.zeros_like(residual),reduction='none',beta=1.).mean(-1)
    rs=rgb_resp['sigma'].detach().float().clamp_min(1e-4)
    ts=tir_resp['sigma'].detach().float().clamp_min(1e-4)
    size_residual=torch.log((s*rs/ts).clamp_min(1e-5)).clamp(-4,4)
    size=F.smooth_l1_loss(size_residual,torch.zeros_like(size_residual),
                          beta=.25,reduction='none').mean(-1)
    if float(overlap_weight):
        rh=rgb_resp['heatmap'].detach().float()
        th=tir_resp['heatmap'].detach().float()
        if rh.shape[-2:]!=th.shape[-2:]:
            th=F.interpolate(th,size=rh.shape[-2:],mode='bilinear',align_corners=True)
        tw=warp_feature_response(th,s,t)
        flat_r=rh.flatten(1);flat_t=tw.flatten(1)
        dice=(2*(flat_r*flat_t).sum(1)+1e-8)/(
            flat_r.square().sum(1)+flat_t.square().sum(1)+1e-8)
        overlap=dice.clamp(0,1)
        overlap_loss=1-overlap
    else:
        overlap=s.new_zeros(len(q));overlap_loss=overlap
    per=center+float(size_weight)*size+float(overlap_weight)*overlap_loss
    total=(per*q).sum()/(q>0).float().sum().clamp_min(1.)
    denom=q.sum().clamp_min(1e-6)
    return total,{
        'warp_center_px':(residual.detach().norm(dim=1)*q).sum()/denom,
        'warp_size_loss':(size.detach()*q).sum()/denom,
        'warp_response_overlap':(overlap.detach()*q).sum()/denom,
        'warp_valid_rate':(q>0).float().mean().detach(),
    }
