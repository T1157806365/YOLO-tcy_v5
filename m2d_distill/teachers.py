"""Frozen RGB/TIR YOLO26m teachers used ONLY during training.

Extract P3/P4/P5 backbone features from the actual one-modal detector layer
indices (the same scale_to_index as the student's YOLO26m backbone).
Teacher models are intentionally NOT registered as student nn.Module children;
no teacher weights or buffers enter student state_dict / optimizer / inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import torch
from torch import nn
from ultralytics import YOLO

from engine.evaluator import postprocess_predictions
from ga_monoalign_pcgrad.teachers import xywh2xyxy, box_iou_single


class FrozenPyramidTeacher:
    def __init__(self, checkpoint: str, scale_to_index: dict, device, nc: int = 1):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Teacher checkpoint missing: {checkpoint}")
        self.net = YOLO(str(checkpoint)).model.to(device).eval()
        self.net.requires_grad_(False)
        if int(getattr(self.net, 'nc', -1)) != int(nc):
            raise ValueError(f"Teacher classes differ from student: {checkpoint}")
        self.scale_to_index = {int(s): int(i) for s, i in scale_to_index.items() if int(s) in (3, 4, 5)}
        if sorted(self.scale_to_index) != [3, 4, 5]:
            raise ValueError(f"Need P3, P4, P5 indices: {scale_to_index}")
        self._features = {}
        self._hooks = []
        for scale, idx in self.scale_to_index.items():
            if idx >= len(self.net.model):
                raise ValueError(f"Teacher backbone has no layer {idx} for P{scale}")
            self._hooks.append(self.net.model[idx].register_forward_hook(self._make_hook(scale)))

    def _make_hook(self, scale):
        def fn(_module, _inputs, output):
            if not torch.is_tensor(output):
                raise TypeError(f"Teacher P{scale} output is not a tensor")
            self._features[scale] = output.detach()
        return fn

    @torch.no_grad()
    def get(self, images, gt_boxes, gt_batch_idx, *, conf=0.30, iou=0.50, microbatch=2):
        self.net.eval()
        n = int(images.shape[0]); height, width = images.shape[-2:]
        acc: Dict[int, list] = {3: [], 4: [], 5: []}
        good = []
        for lo in range(0, n, max(1, int(microbatch))):
            hi = min(lo + max(1, int(microbatch)), n)
            x = images[lo:hi]
            self._features = {}
            with torch.autocast(device_type=x.device.type,
                                dtype=torch.float16,
                                enabled=(x.device.type == 'cuda')):
                prediction = self.net(x)
            if sorted(self._features) != [3, 4, 5]:
                raise RuntimeError(f"Missing teacher pyramid hooks: found {sorted(self._features)}")
            for scale in (3, 4, 5):
                acc[scale].append(self._features[scale].float())
            # Diagnostic teacher correctness is evaluated against EACH modality's
            # own native GT coordinate system; NEVER compare TIR boxes to RGB GT.
            detections = postprocess_predictions(
                prediction, self.net, conf_thres=max(0.001, conf * 0.5),
                iou_thres=0.70, max_det=100)
            for local_j, det in enumerate(detections):
                global_j = lo + local_j
                gt = gt_boxes[gt_batch_idx == global_j]
                correct = False
                if gt.numel():
                    native_gt = xywh2xyxy(gt.float()) * gt.new_tensor([width, height, width, height])
                    for box, score, cls in zip(det['bboxes'], det['conf'], det['cls']):
                        if int(cls) != 0 or float(score) < conf:
                            continue
                        if float(box_iou_single(box.float(), native_gt).max().item()) >= iou:
                            correct = True
                            break
                good.append(correct)
        feats = {s: torch.cat(acc[s], dim=0) for s in (3, 4, 5)}
        return {'features': feats,
                'good': torch.tensor(good, device=images.device, dtype=torch.bool)}

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class M2DTeacherPair:
    """Not an nn.Module on purpose: no teacher storage in student checkpoints."""
    def __init__(self, cfg, scale_to_index, device, nc=1):
        self.cfg = cfg
        self.rgb = FrozenPyramidTeacher(cfg['rgb_weights'], scale_to_index, device, nc)
        self.tir = FrozenPyramidTeacher(cfg['tir_weights'], scale_to_index, device, nc)

    @torch.no_grad()
    def get(self, batch):
        kw = dict(
            conf=float(self.cfg.get('min_conf', 0.3)),
            iou=float(self.cfg.get('min_gt_iou', 0.5)),
            microbatch=int(self.cfg.get('forward_microbatch', 2)),
        )
        return {
            'rgb': self.rgb.get(batch['rgb_semantic_img'], batch['rgb_semantic_bboxes'],
                                batch['rgb_batch_idx'], **kw),
            'tir': self.tir.get(batch['tir_img'], batch['tir_bboxes'],
                                batch['tir_batch_idx'], **kw),
        }

    def close(self):
        self.rgb.close(); self.tir.close()
