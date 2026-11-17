"""Cautious full-image fusion of global predictions and local ROI predictions.

Do not re-NMS the original YOLO26 end-to-end global outputs. For matched ROI
boxes, replace only coordinates while retaining global confidence and class.
This avoids the score-calibration/NMS failure observed in LARA v1.
"""
from __future__ import annotations
import torch


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel()==0 or b.numel()==0:
        return a.new_zeros((len(a),len(b)))
    p1=torch.maximum(a[:,None,:2],b[None,:,:2]); p2=torch.minimum(a[:,None,2:],b[None,:,2:])
    inter=(p2-p1).clamp_min(0).prod(-1)
    aa=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    bb=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(aa[:,None]+bb[None,:]-inter).clamp_min(1e-7)


def empty_predictions(device: torch.device | str):
    return dict(bboxes=torch.zeros((0,4),device=device),conf=torch.zeros(0,device=device),cls=torch.zeros(0,device=device))


def merge_local_predictions(locals_list: list[dict], max_det=300, nms_iou=0.50):
    if not locals_list:
        return empty_predictions('cpu')
    device=locals_list[0]['bboxes'].device
    pieces={k:[p[k] for p in locals_list if p[k].numel()>0] for k in ('bboxes','conf','cls')}
    if not pieces['bboxes']:
        return empty_predictions(device)
    b=torch.cat(pieces['bboxes'],0);s=torch.cat(pieces['conf'],0);c=torch.cat(pieces['cls'],0)
    order=torch.argsort(s,descending=True)
    keep=[]
    while order.numel()>0 and len(keep)<int(max_det):
        idx=int(order[0]);keep.append(idx)
        if len(order)==1:break
        overlap=pairwise_iou(b[idx:idx+1],b[order[1:]])[0]
        same=c[order[1:]]==c[idx]
        order=order[1:][~((overlap>float(nms_iou))&same)]
    ids=torch.tensor(keep,device=device,dtype=torch.long)
    return {'bboxes':b[ids],'conf':s[ids],'cls':c[ids]}


def safe_replace(global_pred: dict, local_pred: dict, *,
                 min_local_conf: float=0.25, min_iou: float=0.12,
                 min_area_ratio: float=0.35, max_area_ratio: float=2.8,
                 max_center_distance: float=1.8, unmatched_add_conf: float=0.90):
    """GT-free replacement, preserving globally ranked scores.

    Matching rules are conservative and must be tuned on val (not test).
    An optional very-high-confidence unmatched ROI prediction can rescue
    images with no global proposals; it is not always added.
    """
    ret={k:v.clone() for k,v in global_pred.items()}
    n=len(ret['bboxes']);m=len(local_pred['bboxes'])
    if m==0:return ret,{'replaced':0,'added':0}
    gb=ret['bboxes'];lb=local_pred['bboxes'];ls=local_pred['conf']
    matched=set();replaced=0
    if n:
        iou=pairwise_iou(gb,lb)
        for gi in torch.argsort(ret['conf'],descending=True).tolist():
            # Try candidates in decreasing IoU, but accept only if they pass rules.
            for li in torch.argsort(iou[gi],descending=True).tolist():
                if li in matched or int(local_pred['cls'][li]) != int(ret['cls'][gi]):
                    continue
                if float(ls[li])<min_local_conf or float(iou[gi,li])<min_iou:
                    continue
                area_g=(gb[gi,2:]-gb[gi,:2]).clamp_min(1).prod()
                area_l=(lb[li,2:]-lb[li,:2]).clamp_min(1).prod()
                ratio=float((area_l/area_g).item())
                if not (min_area_ratio<=ratio<=max_area_ratio):
                    continue
                center_g=(gb[gi,:2]+gb[gi,2:])*0.5
                center_l=(lb[li,:2]+lb[li,2:])*0.5
                diag=torch.linalg.norm((gb[gi,2:]-gb[gi,:2]).clamp_min(1)).clamp_min(1)
                if float(torch.linalg.norm(center_l-center_g)/diag)>max_center_distance:
                    continue
                ret['bboxes'][gi]=lb[li]
                matched.add(li);replaced+=1;break
    # Only rescue empty global detections by default; do not add competing boxes
    # on images with existing global detections.
    added=0
    if not n:
        good=(ls>=unmatched_add_conf).nonzero(as_tuple=False).flatten()
        if len(good):
            li=good[ls[good].argmax()]
            for k in ('bboxes','conf','cls'):
                ret[k]=torch.cat((ret[k],local_pred[k][li:li+1]),0)
            added=1
    return ret,{'replaced':replaced,'added':added}
