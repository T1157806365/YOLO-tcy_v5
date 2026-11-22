"""Student-side GPR-MS KD integration into the existing unified dual-head.

No changes to inference, YOLO heads, local ROI crop, geometry, or alignment.
Student feature hooks work on the exact features already computed by the
original forward, avoiding a second student backbone pass.
"""
from __future__ import annotations
import torch
from .model import GMLSV2DualHeadHybrid, GMLSV2GeoBoxDualHead
from .distillation import MultiScaleKDLoss, kd_weight, SCALES


class GPRMSKDMixin:
    """Mixin on the existing GeoBox or GeoCenter dual-head model."""
    def __init__(self,cfg,*,verbose=True):
        super().__init__(cfg,verbose=verbose)
        d=cfg.get('distillation',{})
        if not d.get('enabled',False):
            raise ValueError('GPR-MS KD model requires distillation.enabled=true')
        if int(cfg['train']['epochs'])!=100:
            raise ValueError('GPR-MS KD schedule requires exactly 100 epochs')
        self._kd_max_weight=float(d.get('max_weight',.01))
        self._kd_active=False
        self._kd_gradient_probe_once=False
        self._kd_student_features={}
        # Holds a plain Python object, NOT a torch.nn.Module. Teacher networks
        # must never enter student.parameters/state_dict/DDP checkpoints.
        self.__dict__['_kd_runtime']=None
        self._kd_hooks=[]
        g=self.global_encoder
        self._kd_indices={s:int(g.scale_to_index[s]) for s in SCALES}
        for modality,branch in [('rgb',g.model),('tir',g.tir_backbone)]:
            for scale in SCALES:
                idx=self._kd_indices[scale]
                if idx>=len(branch) or idx<0:
                    raise RuntimeError(f'Bad {modality} P{scale} backbone index: {idx}')
                def capture(_,__,output,m=modality,k=scale):
                    if self._kd_active:
                        if not torch.is_tensor(output):
                            raise TypeError(f'{m} P{k} student activation is not Tensor')
                        self._kd_student_features[(m,k)]=output
                self._kd_hooks.append(branch[idx].register_forward_hook(capture))
        if verbose:
            print('[GPR-MS KD] native-modality P3 attention / P4 contrast / P5 semantic; '
                  'teachers are training-only',flush=True)

    def prepare_teachers(self,device):
        """Run before DDP wrapping. Fail immediately on absent/incompatible PT."""
        if self.__dict__['_kd_runtime'] is not None:return
        teacher_runtime=MultiScaleKDLoss(self.cfg,self._kd_indices,device)
        self.__dict__['_kd_runtime']=teacher_runtime
        # No teacher parameters are ever registered in our checkpoint.
        if any(k.startswith('_kd_runtime') or k.startswith('teacher')
               for k in self.state_dict()):
            raise AssertionError('Teacher weights unexpectedly appeared in state_dict')
        print('[GPR-MS KD] Both frozen teachers loaded; backbone indices='
              f'{self._kd_indices}; no teacher parameters in student state_dict',flush=True)

    def loss(self,batch):
        if not self.training:
            return super().loss(batch)
        factor=kd_weight(int(getattr(self,'epoch',0)),self._kd_max_weight)
        self._kd_student_features={}
        self._kd_active=factor>0
        try:
            baseline,stats=super().loss(batch)
            stats=dict(stats)
            stats['kd_weight']=float(factor)
            if factor<=0:
                z=baseline.detach().new_zeros(())
                stats.update({'kd_raw':z,'kd_weighted':z,
                              'kd_rgb_valid_rate':0.0,'kd_tir_valid_rate':0.0})
                return baseline,stats
            runtime=self.__dict__['_kd_runtime']
            if runtime is None:
                raise RuntimeError('GPR-MS KD teacher not initialized: call prepare_teachers(device) before training')
            raw,extra=runtime(batch,self._kd_student_features)
            weighted=raw*factor
            if self._kd_gradient_probe_once:
                # Diagnostic only, during SINGLE-GPU --smoke; no DDP collectives,
                # no optimizer changes. Compare gradients w.r.t. the SAME P3
                # activations (not the full parameter-gradient vector).
                act=[self._kd_student_features[(m,3)] for m in ('rgb','tir')]
                bg=torch.autograd.grad(baseline,act,retain_graph=True,allow_unused=True)
                kg=torch.autograd.grad(weighted,act,retain_graph=True,allow_unused=True)
                bs=torch.cat([g.detach().float().flatten() if g is not None else
                              torch.zeros_like(x).flatten() for x,g in zip(act,bg)])
                ks=torch.cat([g.detach().float().flatten() if g is not None else
                              torch.zeros_like(x).flatten() for x,g in zip(act,kg)])
                ratio=ks.norm()/(bs.norm()+1e-12)
                cos=torch.nn.functional.cosine_similarity(bs[None],ks[None],dim=1,eps=1e-12)[0]
                extra['kd_p3_gradient_ratio']=ratio.detach()
                extra['kd_p3_gradient_cosine']=cos.detach()
                self._kd_gradient_probe_once=False
                print(f'[KD GRAD PROBE: P3 activations] ratio={float(ratio):.4g} '
                      f'cos={float(cos):.4g}; diagnostic only',flush=True)
            total=baseline+weighted
            stats.update(extra)
            stats['kd_weighted']=weighted.detach()
            stats['total']=total.detach()
            return total,stats
        finally:
            self._kd_active=False
            self._kd_student_features={}


class GPRMSKDGeoBox(GPRMSKDMixin,GMLSV2GeoBoxDualHead):
    architecture='gmls_v2_dualhead_geobox_gprmskd_p345_100e_v1'


class GPRMSKDGeoCenter(GPRMSKDMixin,GMLSV2DualHeadHybrid):
    architecture='gmls_v2_dualhead_geocenter_gprmskd_p345_100e_v1'


def build_kd_model(cfg,*,verbose=True):
    g=bool(cfg.get('model',{}).get('geo_box',{}).get('enabled',False))
    ac=cfg['loss']['alignment']
    if g:
        if not float(ac.get('lambda_final_size',0))>0 or not float(ac.get('lambda_final_giou',0))>0:
            raise ValueError('GPR-MS KD GeoBox requires both final size and GIoU losses enabled')
        return GPRMSKDGeoBox(cfg,verbose=verbose)
    if any(float(ac.get(k,0))!=0 for k in ('lambda_final_size','lambda_final_giou')):
        raise ValueError('GeoCenter KD variant must not contain GeoBox positive weights')
    return GPRMSKDGeoCenter(cfg,verbose=verbose)
