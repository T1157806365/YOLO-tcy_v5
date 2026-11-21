"""Multi-ROI YOLO label/box transformations and classwise NMS.

Train-only GT functions are separated; inference never reads GT.
"""
import torch
from .geometry import _meta_arrays


def labels_for_crops(batch, geo, k, min_visible=0.6):
    """Labels for B*K crop tensor in image-major order. Returns (batch, positive_rois)."""
    B = batch['rgb_img'].shape[0]
    H,W = batch['rgb_img'].shape[-2:]
    assert geo.xy0.shape == (B*k,2)
    boxes, classes, batch_ids = [], [], []
    positive = 0
    indices = batch['rgb_batch_idx'].long()
    for i in range(B):
        selected = indices == i
        bb = batch['rgb_bboxes'][selected]
        cc = batch['rgb_cls'][selected]
        if bb.numel() == 0:
            continue
        pix = bb*bb.new_tensor([W,H,W,H])
        lo = pix[:,:2]-pix[:,2:]*.5
        hi = pix[:,:2]+pix[:,2:]*.5
        for j in range(k):
            dst = i*k+j
            origin = geo.xy0[dst].detach()
            clipped_lo = (lo-origin).clamp(0,geo.size)
            clipped_hi = (hi-origin).clamp(0,geo.size)
            wh = (clipped_hi-clipped_lo).clamp_min(0)
            vis = wh.prod(-1)/pix[:,2:].prod(-1).clamp_min(1e-8)
            keep = (vis>=float(min_visible))&(wh[:,0]>=1)&(wh[:,1]>=1)
            if bool(keep.any()):
                positive+=1
                boxes.append(torch.cat(((clipped_lo[keep]+clipped_hi[keep])*.5, wh[keep]),1)/geo.size)
                classes.append(cc[keep])
                batch_ids.append(torch.full((int(keep.sum()),),dst,device=bb.device,dtype=torch.long))
    ref=batch['rgb_bboxes'];index=batch['rgb_batch_idx']
    return {
        'rgb_cls':torch.cat(classes) if classes else ref.new_zeros((0,1)),
        'rgb_bboxes':torch.cat(boxes) if boxes else ref.new_zeros((0,4)),
        'rgb_batch_idx':torch.cat(batch_ids) if batch_ids else index.new_zeros((0,))
    },positive


def roi_coverage_union(batch,geo,k,min_visible=0.6):
    """GT only for logging: count objects covered by >=1 of this image's ROIs."""
    B = batch['rgb_img'].shape[0]
    H,W = batch['rgb_img'].shape[-2:]
    index=batch['rgb_batch_idx'].long()
    hit=total=0
    for i in range(B):
        bb=batch['rgb_bboxes'][index==i]
        if bb.numel()==0: continue
        px=bb*bb.new_tensor([W,H,W,H])
        lo=px[:,:2]-px[:,2:]*.5;hi=px[:,:2]+px[:,2:]*.5
        vmax=torch.zeros(len(bb),device=px.device,dtype=px.dtype)
        for j in range(k):
            origin=geo.xy0[i*k+j].detach()
            cl=torch.maximum(lo,origin)
            ch=torch.minimum(hi,origin+geo.size)
            frac=(ch-cl).clamp_min(0).prod(-1)/px[:,2:].prod(-1).clamp_min(1e-8)
            vmax=torch.maximum(vmax,frac)
        total+=len(bb);hit+=int((vmax>=min_visible).sum().item())
    return hit,total


def crop_box_to_semantic(boxes, geo, batch, image_index, roi_index):
    """Convert one ROI's xyxy boxes into RGB-semantic absolute xyxy coordinates."""
    if boxes.numel()==0:return boxes.reshape(0,4)
    r_hi, p_hi = _meta_arrays([batch['rgb_ratio_pad'][image_index]], boxes)
    r_se, p_se = _meta_arrays([batch['rgb_semantic_ratio_pad'][image_index]], boxes)
    high=boxes.reshape(-1,2,2)+geo.xy0[roi_index].detach().view(1,1,2)
    sem=(high-p_hi[0].view(1,1,2))/r_hi[0]*r_se[0]+p_se[0].view(1,1,2)
    h,w=batch['rgb_semantic_img'].shape[-2:]
    sem[...,0]=sem[...,0].clamp(0,w)
    sem[...,1]=sem[...,1].clamp(0,h)
    return sem.reshape(-1,4)


def per_image_merge(local_predictions, geo, batch, k, conf=0.001, nms_iou=0.5,max_det=300):
    """Merge B*K local predictions using ONE shared detector; no global boxes."""
    B = batch['rgb_img'].shape[0]
    results=[]
    for i in range(B):
        chunks=[]
        for j in range(k):
            roi_index=i*k+j
            p=local_predictions[roi_index]
            if len(p['bboxes'])==0: continue
            mapped=crop_box_to_semantic(p['bboxes'],geo,batch,i,roi_index)
            chunks.append((mapped,p['conf'],p['cls']))
        if not chunks:
            empty=local_predictions[0]['bboxes'].new_zeros((0,4))
            results.append({'bboxes':empty,'conf':empty.new_zeros((0,)),
                            'cls':empty.new_zeros((0,))})
            continue
        boxes=torch.cat([c[0] for c in chunks]);scores=torch.cat([c[1] for c in chunks]);classes=torch.cat([c[2] for c in chunks])
        valid=(scores>=float(conf))&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
        boxes,scores,classes=boxes[valid],scores[valid],classes[valid]
        if len(boxes)==0:
            results.append({'bboxes':boxes,'conf':scores,'cls':classes});continue
        # Greedy, class-aware NMS (stable and dependency-free).
        order=scores.argsort(descending=True)
        kept=[]
        while len(order) and len(kept)<int(max_det):
            first=order[0]
            kept.append(first)
            if len(order)==1:break
            others=order[1:]
            a=boxes[first];b=boxes[others]
            inter=(torch.minimum(a[2:],b[:,2:])-torch.maximum(a[:2],b[:,:2])).clamp_min(0).prod(-1)
            areas_a=(a[2:]-a[:2]).clamp_min(0).prod()
            areas_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
            iou=inter/(areas_a+areas_b-inter).clamp_min(1e-7)
            suppress=(iou>float(nms_iou))&(classes[others]==classes[first])
            order=others[~suppress]
        keep=torch.stack(kept)
        results.append({'bboxes':boxes[keep],'conf':scores[keep],'cls':classes[keep]})
    return results
