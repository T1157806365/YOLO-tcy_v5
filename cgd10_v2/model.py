"""CGD-10 V2 student: original 10 GeoBox + constant P3 geometry distillation.

All student modules/state_dict keys are IDENTICAL to GMLSV2GeoBoxDualHead.
The frozen teachers are Python-only training utilities; at inference the model
is loaded by the unchanged GeoBox dual-head benchmark.
"""
from __future__ import annotations
import torch
from gmls_roi_hybrid.model import GMLSV2GeoBoxDualHead
from .teachers import FeatureGeometryTeachers


class CGD10FeatureGeoBox(GMLSV2GeoBoxDualHead):
    architecture=GMLSV2GeoBoxDualHead.architecture

    def __init__(self,cfg,*,verbose=True):
        d=cfg.get('geometry_distillation',{})
        if not d.get('enabled',False) or d.get('mode')!='feature_geometry_v2':
            raise ValueError('Requires geometry_distillation.mode=feature_geometry_v2')
        if not bool(cfg['model']['geo_box'].get('enabled',False)):
            raise ValueError('10 GeoBox must stay enabled')
        if int(cfg['train']['epochs'])!=100 or bool(cfg.get('train',{}).get('stage2',{}).get('enabled',False)):
            raise ValueError('100 epochs one-stage joint training only')
        if cfg.get('distillation',{}).get('enabled',False):
            raise ValueError('Do not combine GPR-MS KD with CGD-10 V2')
        for key in ('route_weight','offset_weight','warp_weight'):
            if float(d.get(key,0))<0:raise ValueError(f'{key} must be nonnegative')
        if not any(float(d.get(key,0))>0 for key in ('route_weight','offset_weight','warp_weight')):
            raise ValueError('All KD weights are zero')
        super().__init__(cfg,verbose=verbose)
        self.__dict__['_cgd_teachers']=None
        self.__dict__['_cgd_info']=None
        self._cgd_gradient_probe_once=False
        self._kd_gradient_probe_once=False
        if verbose:
            print('[CGD-10 V2] Original GeoBox/GTGA/Top-2/dual heads unchanged; '
                  'RGB+TIR P3 response -> Router, offsets, final S,t',flush=True)
            print('[CGD-10 V2] Route/Offset/Warp KD constant from epoch 1 to 100; '
                  'NO KD warmup, decay, or stage2',flush=True)

    def prepare_teachers(self,device):
        if self.__dict__['_cgd_teachers'] is not None:return
        p3=int(self.global_encoder.scale_to_index[3])
        self.__dict__['_cgd_teachers']=FeatureGeometryTeachers(self.cfg,p3,device)
        if any('_cgd_teachers' in k for k in self.state_dict()):
            raise RuntimeError('Teacher was accidentally registered into student state_dict')

    def encode_and_crop(self,batch,*,return_debug=False,train_rois=False):
        ex=super().encode_and_crop(batch,return_debug=return_debug,train_rois=train_rois)
        if self.training and train_rois:
            self.__dict__['_cgd_info']=ex['alignment_info']
        return ex

    def loss(self,batch):
        if not self.training:
            return super().loss(batch)
        self.__dict__['_cgd_info']=None
        try:
            orig,parts=super().loss(batch)
            info=self.__dict__['_cgd_info']
            if info is None:
                raise RuntimeError('Alignment_info missing from 10 GeoBox forward')
            runtime=self.__dict__['_cgd_teachers']
            if runtime is None:
                raise RuntimeError('Frozen teachers not initialized. Need prepare_teachers() before forward')
            with torch.autocast(device_type=orig.device.type,enabled=False):
                route,offset,warp,logs=runtime(batch,info)
                B=int(batch['rgb_img'].shape[0]);d=self.cfg['geometry_distillation']
                rw=B*float(d.get('route_weight',0))*route
                ow=B*float(d.get('offset_weight',0))*offset
                ww=B*float(d.get('warp_weight',0))*warp
                kd=rw+ow+ww
                total=orig.float()+kd
            if self._cgd_gradient_probe_once or self._kd_gradient_probe_once:
                tensors=(info['sampling_scale'],info['sampling_translation'],
                         info['gmls_v2_quality_logits'],info['coarse_offset_gmls'],
                         info['coarse_offset_dmls'])
                names=('scale','translation','route_logits','gmls_offset','dmls_offset')
                active=[(name,t) for name,t in zip(names,tensors)
                        if torch.is_tensor(t) and t.requires_grad]
                grads=torch.autograd.grad(kd,[t for _,t in active],
                                          retain_graph=True,allow_unused=True) if active else []
                results={name:(0.0 if g is None else float(g.detach().float().norm()))
                         for (name,_),g in zip(active,grads)}
                for name in names:
                    print(f'[CGD-10 V2 GRAD] {name} norm={results.get(name,0.0):.6g}',flush=True)
                self._cgd_gradient_probe_once=False
                self._kd_gradient_probe_once=False
            out=dict(parts)
            out.update(logs)
            out.update({'cgd_route_weight':float(d.get('route_weight',0)),
                        'cgd_offset_weight':float(d.get('offset_weight',0)),
                        'cgd_warp_weight':float(d.get('warp_weight',0)),
                        'cgd_route_weighted':rw.detach(),
                        'cgd_offset_weighted':ow.detach(),
                        'cgd_warp_weighted':ww.detach(),
                        'cgd_total_weighted':kd.detach(),
                        'total':total.detach()})
            return total,out
        finally:
            self.__dict__['_cgd_info']=None
