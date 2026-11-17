"""Frozen v5 global HVAD → native RGB high-resolution ROI + aligned TIR ROI.

The global model is ALWAYS evaluated without labels. ROI centers are selected
from predicted boxes, with a GT-free GTGA-anchor fallback when empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import torch

from engine.evaluator import postprocess_predictions
from .geometry import sample_native_hr_and_tir, CropGeometry


@dataclass
class ROIProposal:
    rank: int
    score: float
    rgb: torch.Tensor
    tir: torch.Tensor
    geometry: CropGeometry
    center_sem_xy: tuple[float, float]
    from_detector: bool


@torch.no_grad()
def get_global_results(model, batch: Mapping, conf: float = 0.001, iou: float = 0.70,
                       max_det: int = 300):
    """Return decoded predictions, geometry debug. Requires a GTGA-enabled global model."""
    result = model(
        batch['rgb_img'], batch['tir_img'],
        rgb_semantic=batch['rgb_semantic_img'], return_features=True,
    )
    if not isinstance(result, dict) or 'pred' not in result or 'alignment_info' not in result:
        raise RuntimeError('Global model must return {pred, alignment_info} with return_features=True')
    predictions = postprocess_predictions(result['pred'], model=model, conf_thres=conf,
                                          iou_thres=iou, max_det=max_det)
    info = result['alignment_info']
    if 'sampling_scale' not in info or 'sampling_translation' not in info:
        raise RuntimeError('GTGA alignment_info missing sampling_scale / sampling_translation')
    return predictions, info


def _centers_from_predictions(pred: Mapping, sem_hw: tuple[int, int],
                              topk: int, min_score: float, min_center_dist: float):
    h, w = sem_hw
    boxes = pred['bboxes'].detach().float()
    scores = pred['conf'].detach().float()
    classes = pred['cls'].detach()
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise RuntimeError('postprocess_predictions did not return [N,4] boxes')
    result = []
    for j in torch.argsort(scores, descending=True).tolist():
        if float(scores[j]) < min_score or int(classes[j]) != 0:
            continue
        b = boxes[j]
        cx, cy = float((b[0]+b[2]).item()/2), float((b[1]+b[3]).item()/2)
        if not (0 <= cx <= w and 0 <= cy <= h):
            continue
        if any((cx-px)**2 + (cy-py)**2 < min_center_dist**2 for px,py,_ in result):
            continue
        result.append((cx, cy, float(scores[j].item())))
        if len(result) >= topk:
            break
    return result


@torch.no_grad()
def generate_rois(global_model, batch: Mapping, *, topk: int = 2, roi_size: int = 640,
                  min_score: float = 0.001, min_center_dist: float = 32.0,
                  global_conf: float = 0.001, global_iou: float = 0.70,
                  predictions=None, alignment_info=None):
    """Generate B × K ROIs without ever reading ground truth.

    `batch` must already be preprocessed to [0,1] GPU tensors by the original
    engine.runtime.preprocess_batch. Returns (per_image_proposals, global_preds).
    """
    if predictions is None or alignment_info is None:
        predictions, alignment_info = get_global_results(global_model, batch,
                                                          conf=global_conf, iou=global_iou)
    bsz = int(batch['rgb_img'].shape[0])
    sem_hw = tuple(batch['rgb_semantic_img'].shape[-2:])
    scale = alignment_info['sampling_scale']
    translation = alignment_info['sampling_translation']
    anchors = alignment_info.get('coarse_anchor', None)
    groups = []
    for i in range(bsz):
        centers = _centers_from_predictions(predictions[i], sem_hw, int(topk),
                                            float(min_score), float(min_center_dist))
        if not centers:
            if anchors is not None:
                a = anchors[i].detach().float().reshape(2).clamp(0, 1)
                centers = [(float(a[0]) * (sem_hw[1]-1), float(a[1]) * (sem_hw[0]-1), 0.0)]
            else:
                centers = [((sem_hw[1]-1)*0.5, (sem_hw[0]-1)*0.5, 0.0)]
        proposals=[]
        for rank,(cx,cy,score) in enumerate(centers):
            rgb_roi,tir_roi,geom=sample_native_hr_and_tir(
                batch['rgb_img'][i],batch['tir_img'][i],(cx,cy),sem_hw,
                batch['rgb_ratio_pad'][i],batch['rgb_semantic_ratio_pad'][i],
                scale[i],translation[i],roi_size=roi_size,
            )
            proposals.append(ROIProposal(rank, score, rgb_roi, tir_roi, geom,
                                         (cx,cy),score > 0.0))
        groups.append(proposals)
    return groups, predictions
