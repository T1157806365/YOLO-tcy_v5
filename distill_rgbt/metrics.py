"""Optional old-style mono/geometry validation, with alignment-off support."""
from __future__ import annotations
import torch
from torch import nn
from engine.evaluator import validate_metrics
from engine.runtime import preprocess_batch, autocast_context
from .teachers import xywh_to_xyxy, box_iou_single


class MonoEval(nn.Module):
    def __init__(self, student, modality):
        super().__init__()
        self.nc, self.names, self.end2end = student.nc, student.names, student.end2end
        self.model_name, self.modality = student.model_name, modality
        object.__setattr__(self, 'student', student)

    def forward(self, rgb, tir=None, rgb_semantic=None, **kwargs):
        if self.modality == 'rgb':
            return self.student.aux_predict('rgb', rgb_img=rgb_semantic)
        return self.student.aux_predict('tir', tir_img=tir)


class TirGTLoader:
    def __init__(self, loader): self.loader = loader
    def __len__(self): return len(self.loader)
    def __iter__(self):
        for b in self.loader:
            b = dict(b)
            b['rgb_semantic_bboxes'] = b['tir_bboxes']
            b['rgb_batch_idx'] = b['tir_batch_idx']
            b['rgb_cls'] = b['tir_cls']
            yield b


@torch.no_grad()
def mono_metrics(student, loader, device, amp):
    if not student.mono_enabled: return {}
    r = validate_metrics(MonoEval(student,'rgb'), loader, device, amp=amp,
                         conf_thres=.001, iou_thres=.7, max_det=300, preview_num=0)
    t = validate_metrics(MonoEval(student,'tir'), TirGTLoader(loader), device, amp=amp,
                         conf_thres=.001, iou_thres=.7, max_det=300, preview_num=0)
    return ({'rgb_only_'+k:float(v) for k,v in r.items() if isinstance(v,(int,float))}
            | {'tir_only_'+k:float(v) for k,v in t.items() if isinstance(v,(int,float))})


@torch.no_grad()
def geometry_iou(student, loader, device, max_images=96):
    if not student.core.alignment_enabled:
        return {'align_iou':float('nan'), 'align_zero':0,
                'align_pairs':0, 'align_images_scanned':0}
    student.eval()
    results=[]; scanned=0; zero=0
    for batch in loader:
        batch=preprocess_batch(batch,device)
        with autocast_context(device,False):
            pred=student.core.predict(batch['rgb_img'],batch['tir_img'],
                rgb_semantic=batch['rgb_semantic_img'],return_features=True,
                return_alignment_debug=True)
        info=pred['alignment_info']
        s=info['sampling_scale'].float(); t=info['sampling_translation'].float()
        bsz=batch['rgb_semantic_img'].shape[0]
        for i in range(bsz):
            rb=batch['rgb_semantic_bboxes'][batch['rgb_batch_idx']==i]
            tb=batch['tir_bboxes'][batch['tir_batch_idx']==i]
            if rb.shape[0]!=1 or tb.shape[0]!=1:continue
            scale=s[i].flatten()[:2].clamp_min(1e-6)
            shift=t[i].flatten()[:2]
            mapped=torch.cat(((tb[0,:2]-shift)/scale,tb[0,2:]/scale))
            box=xywh_to_xyxy(mapped[None])[0]
            iou=float(box_iou_single(box,xywh_to_xyxy(rb)).max().item())
            results.append(iou);zero+=int(iou<=1e-12)
        scanned+=bsz
        if scanned>=max_images:break
    return {'align_iou':sum(results)/len(results) if results else float('nan'),
            'align_zero':zero,'align_pairs':len(results),'align_images_scanned':scanned}
