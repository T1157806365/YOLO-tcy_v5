"""Main-branch-compatible RGB/TIR teacher initialization and reliable ROI feature KD.

The teacher models deliberately live OUTSIDE the student's torch.nn.Module tree.
Only the student original RGB/TIR backbone P3 outputs receive distillation loss.
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn


def load_yolo_teacher(path, nc=1):
    from ultralytics import YOLO
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    net = YOLO(str(path)).model.cpu().eval()
    if int(net.nc) != int(nc):
        raise ValueError(f"Teacher classes {net.nc} != student classes {nc} at {path}")
    if not hasattr(net, 'model') or not isinstance(net.model, (nn.ModuleList, nn.Sequential)):
        raise TypeError(f"Unexpected teacher layer structure: {type(getattr(net, 'model', None))}")
    return net


def _copy_backbone(student_layers, teacher_layers, label):
    a = nn.ModuleList(list(student_layers))
    b = nn.ModuleList(list(teacher_layers))
    if len(a) != len(b):
        raise ValueError(f"{label} backbone length differs: {len(a)} != {len(b)}")
    for idx, (dst, src) in enumerate(zip(a, b)):
        if type(dst) is not type(src) or getattr(dst, 'f', None) != getattr(src, 'f', None):
            raise ValueError(f"{label} architecture differs at layer {idx}")
    sa, sb = a.state_dict(), b.state_dict()
    if set(sa) != set(sb):
        raise ValueError(f"{label} state_dict keys differ")
    for key in sa:
        if sa[key].shape != sb[key].shape:
            raise ValueError(f"{label}: shape differs for {key}: {sa[key].shape} vs {sb[key].shape}")
    a.load_state_dict(sb, strict=True)
    # Check the copy really occurred, including BN running statistics.
    for key, tensor in a.state_dict().items():
        if not torch.equal(tensor.cpu(), sb[key].cpu()):
            raise RuntimeError(f"{label}: copy verification failed: {key}")
    print(f"[INIT] {label}: copied {sum(x.numel() for x in a.parameters()):,} backbone parameters", flush=True)


def initialize_backbones(student, cfg):
    weights = cfg['distill']['teachers']
    n = int(student.backbone_len)
    rgb = load_yolo_teacher(weights['rgb'], student.nc)
    _copy_backbone(list(student.model)[:n], list(rgb.model)[:n], 'RGB')
    del rgb
    tir = load_yolo_teacher(weights['tir'], student.nc)
    _copy_backbone(list(student.tir_backbone), list(tir.model)[:n], 'TIR')
    del tir
    print('[INIT] RGB/TIR teacher BACKBONES only. Fusion, alignment, BHLR and neck/head remain original initialization.', flush=True)


def xywh_to_xyxy(x):
    return torch.stack((x[..., 0] - x[..., 2] / 2,
                        x[..., 1] - x[..., 3] / 2,
                        x[..., 0] + x[..., 2] / 2,
                        x[..., 1] + x[..., 3] / 2), -1)


def box_iou_single(box, boxes):
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    a = box.reshape(1, 4).float()
    b = boxes.reshape(-1, 4).float()
    wh = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0)
    inter = wh.prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp_min(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp_min(0).prod(-1)
    return inter / (area_a + area_b - inter).clamp_min(1e-7)


class SingleTeacher:
    def __init__(self, weights, p3_index, device, nc=1):
        self.net = load_yolo_teacher(weights, nc=nc).to(device).eval()
        self.net.requires_grad_(False)
        self.feature = None
        self.hook = self.net.model[int(p3_index)].register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError('Teacher P3 hook did not return a tensor')
        self.feature = output.detach()

    @torch.no_grad()
    def get(self, imgs, boxes, batch_idx, p3_hw, min_conf=.30, min_iou=.50, microbatch=2):
        from engine.evaluator import postprocess_predictions
        n = int(imgs.shape[0])
        h, w = imgs.shape[-2:]
        feats, chosen = [], []
        for st in range(0, n, max(1, int(microbatch))):
            en = min(n, st + max(1, int(microbatch)))
            self.feature = None
            with torch.autocast(device_type=imgs.device.type, dtype=torch.float16,
                                enabled=(imgs.device.type == 'cuda')):
                raw = self.net(imgs[st:en])
            ft = self.feature
            if ft is None:
                raise RuntimeError('Teacher P3 hook was not invoked')
            ft = ft.float()
            if tuple(ft.shape[-2:]) != tuple(p3_hw):
                ft = F.interpolate(ft, size=tuple(p3_hw), mode='bilinear', align_corners=False)
            feats.append(ft)
            dets = postprocess_predictions(raw, self.net,
                                           conf_thres=max(0.001, min_conf * 0.5),
                                           iou_thres=.70, max_det=100)
            for j, det in enumerate(dets):
                i = st + j
                gt = boxes[batch_idx == i]
                if gt.numel() == 0:
                    chosen.append(False)
                    continue
                gt_xyxy = xywh_to_xyxy(gt.float()) * gt.new_tensor((w, h, w, h))
                reliable = False
                for pred_box, score, cls in zip(det['bboxes'], det['conf'], det['cls']):
                    if int(cls) != 0 or float(score) < float(min_conf):
                        continue
                    if float(box_iou_single(pred_box, gt_xyxy).max()) >= float(min_iou):
                        reliable = True
                        break
                chosen.append(reliable)
        return {'p3': torch.cat(feats, 0),
                'good': torch.tensor(chosen, dtype=torch.bool, device=imgs.device)}


class DualTeachers:
    def __init__(self, cfg, p3_index, device, nc=1):
        self.cfg = cfg
        self.rgb = SingleTeacher(cfg['rgb'], p3_index, device, nc=nc)
        self.tir = SingleTeacher(cfg['tir'], p3_index, device, nc=nc)

    def get(self, batch, p3_hw):
        p = self.cfg
        kw = dict(p3_hw=p3_hw,
                  min_conf=float(p.get('min_conf', .3)),
                  min_iou=float(p.get('min_gt_iou', .5)),
                  microbatch=int(p.get('microbatch', 2)))
        rgb = self.rgb.get(batch['rgb_semantic_img'], batch['rgb_semantic_bboxes'],
                           batch['rgb_batch_idx'], **kw)
        tir = self.tir.get(batch['tir_img'], batch['tir_bboxes'],
                           batch['tir_batch_idx'], **kw)
        return rgb, tir


def roi_mask(boxes, batch_idx, bsz, h, w, device, pad=3):
    """GT ROI of corresponding native modality; 3-cell padding for tiny UAVs."""
    mask = torch.zeros(bsz, h, w, dtype=torch.bool, device=device)
    for box, bi in zip(boxes.detach().float().cpu(), batch_idx.detach().long().cpu()):
        i = int(bi)
        cx, cy, bw, bh = [float(t) for t in box]
        x1 = max(0, int((cx - bw / 2) * w) - pad)
        x2 = min(w, int((cx + bw / 2) * w) + pad + 1)
        y1 = max(0, int((cy - bh / 2) * h) - pad)
        y2 = min(h, int((cy + bh / 2) * h) + pad + 1)
        mask[i, y1:y2, x1:x2] = True
    return mask


def spatial_attention(x):
    z = x.float().abs().mean(1)
    mu = z.mean((-2, -1), keepdim=True)
    sd = z.std((-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
    return (z - mu) / sd


def roi_attention_kd(student, teacher, good, boxes, batch_idx):
    b, _c, h, w = student.shape
    if teacher.shape[0] != b:
        raise ValueError('Teacher/student batch mismatch')
    mask = roi_mask(boxes, batch_idx, b, h, w, student.device)
    s = spatial_attention(student)
    t = spatial_attention(teacher).detach()
    losses = []
    for i in range(b):
        if not bool(good[i]) or int(mask[i].sum()) < 2:
            continue
        si, ti = s[i][mask[i]], t[i][mask[i]]
        losses.append(F.kl_div(F.log_softmax(si, 0), F.softmax(ti, 0), reduction='sum'))
    return torch.stack(losses).mean() if losses else s.sum() * 0.0
