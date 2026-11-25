"""Original CDGR + only P3 replacement; no old model checkpoints."""
from __future__ import annotations
import torch
from torch import nn
from cdgr_geobox.model import CDGRGeoBoxDualHead
from .fusion import P2Context,THRFV2P3
from .geometry import roi_cdgr_descriptor
from .loss import local_thermal_targets,heat_loss

class CDGRTHRFV2(CDGRGeoBoxDualHead):
    architecture='cdgr_geobox_thrf_v2_phase_preserving_p3_v1'
    def __init__(self,cfg,*,verbose=True):
        m=cfg['model'];v=m['thrf_v2'];assert v['enabled'] and m['fusion']=='add'
        for key in ('tir_guided_hr_fusion','garf_local_fusion','lcmf_local_fusion'):
            if m.get(key,{}).get('enabled'):raise ValueError('Disable previous fusion variant: '+key)
        super().__init__(cfg,verbose=verbose)
        if list(self.detector.fusion_indices)!=[4,6,10]:raise ValueError('Expected [4,6,10]')
        for i in (4,6,10):
            f=self.detector.fusions[str(i)]
            if type(f).__name__!='AddFusion' or not isinstance(f.tir_proj,nn.Identity):raise ValueError('Expected original Add')
        self._v2_ctx=P2Context();self._v2_ex=None
        self.detector.model[2].register_forward_hook(self._capture_rgb)
        self.detector.tir_backbone[2].register_forward_hook(self._capture_tir)
        args={k:v[k] for k in ('p2_channels','p3_channels','key_dim','value_dim','radius','temperature','geo_sigma','groups','detail_init')}
        self.detector.fusions['4']=THRFV2P3(self._v2_ctx,**args)
        self._v2_weight=float(v['lambda_tir_heat']);self._v2_minvalid=float(v.get('min_tir_valid',.7))
        if not 0<self._v2_weight<=1 or not 0<=self._v2_minvalid<=1:raise ValueError('Invalid supervision settings')
        if verbose:
            n=sum(p.numel() for p in self.detector.fusions['4'].parameters())
            print(f'[THRF-V2] P3 joint main-path; P4/P5 ADD unchanged; added params={n:,}',flush=True)
            print('[THRF-V2] Q=TIR P2/P3; K,V=RGB native P2; preserve TL/TR/BL/BR; actual TIR GT',flush=True)
    def _capture_rgb(self,module,args,out):self._v2_ctx.rgb=out
    def _capture_tir(self,module,args,out):self._v2_ctx.tir=out
    def encode_and_crop(self,batch,*,return_debug=False,train_rois=False):
        ex=super().encode_and_crop(batch,return_debug=return_debug,train_rois=train_rois)
        c=self.cfg['model']['geometry_refinement']
        self._v2_ctx.geo=roi_cdgr_descriptor(batch,ex,center_norm=float(c.get('max_center_shift_semantic_px',24)),log_norm=float(c.get('max_log_scale',.5)),detach=True)
        if train_rois:self._v2_ex=ex
        return ex
    def _clear_v2(self):self._v2_ctx.clear();self._v2_ex=None
    def loss(self,batch):
        try:
            base,parts=super().loss(batch)
            parts=dict(parts);heat=self._v2_ctx.heat
            if heat is None or self._v2_ex is None:raise RuntimeError('THRF V2 did not execute')
            boxes,idx,valid=local_thermal_targets(batch,self._v2_ex,self.min_visible,self._v2_minvalid)
            aux,stats=heat_loss(heat,boxes,idx,valid)
            weighted=aux*self._v2_weight*batch['rgb_img'].shape[0]
            base=base+weighted
            parts.update({f'thrf2_{k}':v.detach() for k,v in self.detector.fusions['4'].last_stats.items()})
            parts.update({f'thrf2_{k}':v.detach() for k,v in stats.items()})
            parts.update(thrf2_tir_heat_loss=aux.detach(),thrf2_tir_heat_weighted=weighted.detach(),total=base.detach())
            return base,parts
        finally:self._clear_v2()
    @torch.no_grad()
    def predict(self,batch,*,conf=.001,iou=.70):
        try:return super().predict(batch,conf=conf,iou=iou)
        finally:self._clear_v2()
