"""Local-only progressive cross-modal encoding on the original CDGR model."""
from types import MethodType
import torch
from torch import nn
from cdgr_geobox.model import CDGRGeoBoxDualHead
from .fusion import ProgressiveComplement
from .geometry import roi_guidance

from .progressive import progressive_features

class CDGRGPCF(CDGRGeoBoxDualHead):
    architecture='cdgr_geobox_gpcf_v1'
    def __init__(self,cfg,*,verbose=True):
        c=cfg['model']['gpcf']
        for key in ('shared_dsbn','mcf','scale_matched_fusion','thrf_v2','tir_guided_hr_fusion','garf_local_fusion','lcmf_local_fusion'):
            if cfg['model'].get(key,{}).get('enabled'):raise ValueError('GPCF requires clean CDGR baseline: '+key)
        super().__init__(cfg,verbose=verbose)
        det=self.detector
        if self.roi_size!=640 or self.num_rois!=2 or list(det.fusion_indices)!=[4,6,10] or det.backbone_len!=11 or len(det.tir_backbone)!=11:raise ValueError('Expected original CDGR YOLO26m Top-2')
        for i in det.fusion_indices:
            if type(det.fusions[str(i)]).__name__!='AddFusion' or not isinstance(det.fusions[str(i)].tir_proj,nn.Identity):raise ValueError('Expected original Add')
        self.gpcf_indices=tuple(c['indices']);self.geometry_enabled=bool(c['geometry'])
        if not self.gpcf_indices or not set(self.gpcf_indices).issubset({4,6}) or len(set(self.gpcf_indices))!=len(self.gpcf_indices):raise ValueError('Only unique P3/P4 feedback supported')
        det.gpcf_blocks=nn.ModuleDict({str(i):ProgressiveComplement(512,int(c['dim']),bool(c['directional']),self.geometry_enabled,float(c['gain'])) for i in self.gpcf_indices})
        det._gpcf_guides={};det._gpcf_keep=None;det.gpcf_intervention='normal'
        # Bind only the local detector, not global encoder or original factory types.
        det.forward_features=MethodType(progressive_features,det)
        self.architecture='cdgr_geobox_gpcf_'+str(c['group']).lower()+'_v1'
        if verbose:print(f'[GPCF] {self.architecture} P3/P4 feedback indices={self.gpcf_indices}; directional={c["directional"]}; geometry={self.geometry_enabled}; extra params={sum(p.numel() for p in det.gpcf_blocks.parameters()):,}',flush=True)
    def _clear_gpcf(self):
        self.detector._gpcf_guides={};self.detector._gpcf_keep=None
    def encode_and_crop(self,batch,*,return_debug=False,train_rois=False):
        ex=super().encode_and_crop(batch,return_debug=return_debug,train_rois=train_rois)
        det=self.detector;self._clear_gpcf()
        if self.geometry_enabled and det.gpcf_intervention not in ('off','no_geometry','rgb_only','tir_only'):
            det._gpcf_guides={i:roi_guidance(batch,ex,(80,80) if i==4 else (40,40)) for i in self.gpcf_indices}
            if train_rois and self.cfg['data'].get('ignore_rgb_empty_tir_positive',False):
                b=len(batch['rgb_img']);k=ex['rois_per_image']
                r=torch.bincount(batch['rgb_batch_idx'].long().flatten(),minlength=b)>0
                t=torch.bincount(batch['tir_batch_idx'].long().flatten(),minlength=b)>0
                det._gpcf_keep=(r|~t).repeat_interleave(k).nonzero(as_tuple=True)[0]
        return ex
    def loss(self,batch):
        for block in self.detector.gpcf_blocks.values():block.collect_stats=True;block.last_stats={}
        try:
            loss,parts=super().loss(batch);parts=dict(parts)
            for i in self.gpcf_indices:
                block=self.detector.gpcf_blocks[str(i)]
                parts.update({f'gpcf_p{3 if i==4 else 4}_{k}':v for k,v in block.last_stats.items()})
            return loss,parts
        finally:
            self._clear_gpcf()
            for block in self.detector.gpcf_blocks.values():block.collect_stats=False
    @torch.no_grad()
    def predict(self,batch,*,conf=.001,iou=.70):
        try:return super().predict(batch,conf=conf,iou=iou)
        finally:self._clear_gpcf()
    def set_intervention(self,mode):
        if mode not in ('normal','off','no_geometry','no_t_to_r','no_r_to_t','rgb_only','tir_only'):raise ValueError(mode)
        self.detector.gpcf_intervention=mode
        for block in self.detector.gpcf_blocks.values():block.intervention=mode
