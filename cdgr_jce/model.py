"""CDGR-JCE: independent baseline extension, local path only."""
from types import MethodType
import torch
from torch import nn
from cdgr_geobox.model import CDGRGeoBoxDualHead
from .fusion import JointEncoder,JointFeedback
from .progressive import progressive_features
from .robust import subset_labels,freeze_bn_statistics


class CDGRJCE(CDGRGeoBoxDualHead):
    architecture='cdgr_geobox_jce_v1'
    def __init__(self,cfg,*,verbose=True):
        c=cfg['model']['jce'];m=cfg['model']
        for key in ('gpcf','shared_dsbn','mcf','scale_matched_fusion','thrf_v2','tir_guided_hr_fusion','garf_local_fusion','lcmf_local_fusion'):
            if m.get(key,{}).get('enabled'):raise ValueError('JCE requires clean CDGR: '+key)
        super().__init__(cfg,verbose=verbose);det=self.detector
        if self.roi_size!=640 or self.num_rois!=2 or list(det.fusion_indices)!=[4,6,10] or det.backbone_len!=11 or len(det.tir_backbone)!=11:raise ValueError('Expected original CDGR YOLO26m Top-2')
        for i in det.fusion_indices:
            f=det.fusions[str(i)]
            if type(f).__name__!='AddFusion' or not isinstance(f.tir_proj,nn.Identity):raise ValueError('Expected original AddFusion')
        dim=int(c['dim']);groups=int(c['groups']);indices=tuple(c['feedback_indices'])
        if not set(indices).issubset({4,6}) or len(set(indices))!=len(indices):raise ValueError('Only P3/P4 feedback')
        det.jce_blocks=nn.ModuleDict({str(i):JointEncoder(512,dim,groups) for i in det.fusion_indices})
        det.jce_feedback=nn.ModuleDict({str(i):JointFeedback(512,groups) for i in indices})
        det.jce_intervention='normal';det.jce_weak_rgb=False
        det.forward_features=MethodType(progressive_features,det)
        self.robust_p=float(c['robust_p']);self.robust_lambda=float(c['robust_lambda']);self.robust_warmup=int(c['robust_warmup'])
        if not 0<=self.robust_p<=1 or not 0<=self.robust_lambda<=1 or self.robust_warmup<0:raise ValueError('Invalid robust options')
        self.architecture='cdgr_geobox_jce_'+str(c['group']).lower()+'_v1'
        if verbose:
            print(f'[JCE] {self.architecture} main fusion P3/P4/P5, feedback={indices}; dim={dim}, groups={groups}, robust_p={self.robust_p}, robust_lambda={self.robust_lambda}',flush=True)
            print('[JCE] No geometry guide, no fixed 0.1 gain, no Mono Head, no QKV; original CDGR and 640/640 ROI',flush=True)
            print('[JCE] extra parameters=',sum(p.numel() for mod in (det.jce_blocks,det.jce_feedback) for p in mod.parameters()),flush=True)
    def set_intervention(self,mode):
        if mode not in ('normal','add','no_feedback','no_interaction','rgb_only','tir_only','zero_rgb','zero_tir'):raise ValueError(mode)
        self.detector.jce_intervention=mode
    def loss(self,batch):
        det=self.detector;mods=[*det.jce_blocks.values(),*det.jce_feedback.values()]
        for m in mods:m.collect_stats=True;m.last_stats={}
        capture=[];hook=None
        # Capture exactly the batch used by the original detector (including optional empty-GT filtering).
        if self.robust_p>0 and self.robust_lambda>0 and self.training:
            def remember(mod,args):
                if args and isinstance(args[0],dict):capture.append(args[0])
            hook=det.register_forward_pre_hook(remember)
        try:
            try:total,parts=super().loss(batch)
            finally:
                if hook is not None:hook.remove()
            parts=dict(parts)
            for i in det.fusion_indices:
                prefix=f'jce_p{ {4:3,6:4,10:5}[i]}_'
                block=det.jce_blocks[str(i)]
                for k in ('delta_over_add','rgb_weight','tir_weight','interaction_weight','fused_abs'):
                    parts[prefix+k]=block.last_stats.get(k,0.)
                if str(i) in det.jce_feedback:
                    fb=det.jce_feedback[str(i)]
                    for k in ('t_to_r_ratio','r_to_t_ratio'):parts[prefix+k]=fb.last_stats.get(k,0.)
            for m in mods:m.collect_stats=False
            raw=total.new_zeros(());weighted=raw;frac=0.;num=0
            if capture:
                if len(capture)!=1:raise RuntimeError('Expected one native local detection batch')
                labels=capture[0];n=len(labels['rgb_img'])
                # Positive ROI only: do not create extra false-negative supervision on empty/unknown labels.
                positive=torch.bincount(labels['rgb_batch_idx'].long().flatten(),minlength=n)>0
                ids=(positive & (torch.rand(n,device=positive.device)<self.robust_p)).nonzero(as_tuple=True)[0]
                num=len(ids);frac=num/max(n,1)
                if num:
                    weak=subset_labels(labels,ids)
                    det.jce_weak_rgb=True
                    with freeze_bn_statistics(det):
                        loss,_=det(weak)
                    det.jce_weak_rgb=False
                    raw=loss.sum()/float(self.num_rois)
                    ramp=min(1.,(float(self.epoch)+1)/max(self.robust_warmup,1)) if self.robust_warmup else 1.
                    weighted=raw*self.robust_lambda*ramp;total=total+weighted
            parts.update(jce_robust_raw=raw.detach(),jce_robust_weighted=weighted.detach(),jce_robust_frac=frac,jce_robust_rois=float(num),total=total.detach())
            return total,parts
        finally:
            det.jce_weak_rgb=False;capture.clear()
            for m in mods:m.collect_stats=False
