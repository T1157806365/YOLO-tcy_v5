"""Standalone frozen-dual-teacher selective KD student for YOLO-tcy_v5/main.

NOT an import, subclass, or patch of online_rgbt_kd_v1 / rgb_preserve_independent_v1.
Teacher weights are held outside the registered torch module tree, so the
student .pt stores only student weights. They are not needed at inference.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from models.rgbt_rlsfa_gtga_centered_bhlr_model import RGBTRLSFAGTGACenteredBHLRV2DetectionModel
from .teachers import FrozenDualTeachers,xywh_to_xyxy,pair_iou


def iou_giou_xyxy(a,b,eps=1e-7):
    lt=torch.maximum(a[...,:2],b[...,:2]); rb=torch.minimum(a[...,2:],b[...,2:])
    inter=(rb-lt).clamp_min(0).prod(-1)
    aa=(a[...,2:]-a[...,:2]).clamp_min(0).prod(-1)
    bb=(b[...,2:]-b[...,:2]).clamp_min(0).prod(-1)
    union=aa+bb-inter
    iou=inter/union.clamp_min(eps)
    outer=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1)
    return iou,iou-(outer-union)/outer.clamp_min(eps)


class ResponseHead(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(c,32,3,padding=1,bias=False),nn.GroupNorm(8,32),
                                nn.SiLU(),nn.Conv2d(32,1,1))
    def forward(self,x):return self.conv(x)


class DualTeacherSelectiveKD(RGBTRLSFAGTGACenteredBHLRV2DetectionModel):
    def __init__(self,*args,selective_cfg=None,teacher_cfg=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.kcfg=dict(selective_cfg or {})
        self.teacher_cfg=dict(teacher_cfg or {})
        p3=int(self.scale_to_index[3])
        dims=self._infer_any_layer_channels((p3,))[p3]
        cr,ct=int(dims['rgb']),int(dims['tir'])
        if cr!=ct:raise RuntimeError(f'P3 teacher/channel expected equal RGB/TIR: {cr}/{ct}')
        self.rgb_adapter=nn.Conv2d(cr,cr,1,bias=False)
        self.tir_adapter=nn.Conv2d(ct,ct,1,bias=False)
        self.fused_adapter=nn.Conv2d(cr,cr,1,bias=False)
        self.tir_projection=nn.Conv2d(ct,cr,1,bias=False)
        for m in (self.rgb_adapter,self.tir_adapter,self.fused_adapter,self.tir_projection):nn.init.dirac_(m.weight)
        self.rgb_response=ResponseHead(cr)
        self.fused_response=ResponseHead(cr)
        self.tir_response=ResponseHead(ct)
        self.quality=nn.Sequential(nn.Linear(10,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.quality[-1].weight);nn.init.zeros_(self.quality[-1].bias)
        self.residual_alpha=nn.Parameter(torch.zeros(()))
        # The original unconditional P3/P4/P5 ADD layers must not be trainable.
        for m in self.fusions.values():m.requires_grad_(False)
        # Deliberately NOT registered as torch module or state_dict entry.
        object.__setattr__(self,'_frozen_dual_teachers',None)

    def attach_frozen_teachers(self,device):
        teachers=self._frozen_dual_teachers
        if teachers is None:
            from .teachers import FrozenDualTeachers
            teachers=FrozenDualTeachers(self.teacher_cfg['rgb_weights'],self.teacher_cfg['tir_weights'],
                                        p3_layer=int(self.scale_to_index[3]))
            object.__setattr__(self,'_frozen_dual_teachers',teachers)
        return teachers.to(device)

    def fuse_backbone_features(self,rgb_outputs,tir_outputs):
        # Only add aligned TIR at P3 inside forward_features() below.
        return list(rgb_outputs),{}

    @staticmethod
    def _vec(x,bi,device,default=(0.,0.)):
        if x is None:return torch.tensor(default,device=device,dtype=torch.float32)
        return x[bi].float().reshape(-1)[:2]

    def _quality_input(self,info,b):
        device=info['sampling_scale'].device
        rows=[]
        for i in range(b):
            sc=self._vec(info.get('sampling_scale'),i,device,(1,1))
            tr=self._vec(info.get('sampling_translation'),i,device)
            coarse=self._vec(info.get('coarse_offset'),i,device)/20.
            conf=info.get('coarse_confidence'); fine=info.get('fine_confidence')
            c=conf[i].float().reshape(-1)[:1] if conf is not None else sc.new_zeros(1)
            fc=fine[i].float().reshape(-1)[:1] if fine is not None else sc.new_zeros(1)
            anchor=self._vec(info.get('coarse_anchor'),i,device,(.5,.5))
            rows.append(torch.cat((sc,tr,coarse,c,fc,anchor)))
        return torch.stack(rows).detach()

    def forward_features(self,*args,**kwargs):
        out=super().forward_features(*args,**kwargs)
        p3=int(self.scale_to_index[3])
        rgb=out['rgb_backbone'][p3]  # post BHLR
        tir=out['aligned_tir_backbone'][p3]
        logits=self.quality(self._quality_input(out['alignment_info'],rgb.shape[0]))
        q=logits.sigmoid().view(-1,1,1,1)
        fused=rgb+torch.tanh(self.residual_alpha)*q*self.tir_projection(tir)
        outputs=list(out['rgb_backbone']);outputs[p3]=fused
        out['fused_backbone']=outputs
        out['fusion_features']={p3:fused}
        out['dual_teacher_quality_logit']=logits
        out['dual_teacher_quality']=q
        return out

    @staticmethod
    def _dense_response_loss(logits,target):
        bce=F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none')
        weight=1.+20.*target
        return (bce*weight).mean()

    @staticmethod
    def _masked_feature_loss(student,teacher,mask,scores):
        """ROI-focused normalized feature KD, selected only for verified teachers."""
        if student.shape!=teacher.shape:
            raise RuntimeError(f'Teacher/student P3 feature mismatch: {student.shape} != {teacher.shape}')
        a=F.normalize(student.float(),dim=1,eps=1e-5)
        b=F.normalize(teacher.detach().float(),dim=1,eps=1e-5)
        dist=(a-b).square().mean(dim=1,keepdim=True)
        valid=mask.float().detach()*scores.detach().float().view(-1,1,1,1)
        return (dist*valid).sum() / valid.sum().clamp_min(1.)

    def _geometry(self,batch,info):
        """Final TIR->RGB geometry for unambiguous native GT pairs."""
        B=int(batch['rgb_semantic_img'].shape[0]); sc=info['sampling_scale'].float()
        tr=info['sampling_translation'].float()
        q=[]; ctr=[]; giou=[]
        for i in range(B):
            ri=batch['rgb_batch_idx']==i; ti=batch['tir_batch_idx']==i
            if int(ri.sum())!=1 or int(ti.sum())!=1:
                q.append(sc.new_zeros(()));continue
            rb=batch['rgb_semantic_bboxes'][ri][0].float();tb=batch['tir_bboxes'][ti][0].float()
            s=sc[i].reshape(-1)[:2].clamp_min(1e-5);t=tr[i].reshape(-1)[:2]
            mapped=torch.cat(((tb[:2]-t)/s,tb[2:]/s),dim=0)
            iou,gio=iou_giou_xyxy(xywh_to_xyxy(mapped),xywh_to_xyxy(rb))
            q.append(iou.detach().clamp(0,1))
            ctr.append(F.smooth_l1_loss((mapped[:2]-rb[:2])*80.,torch.zeros_like(s),beta=1.))
            giou.append(1.-gio)
        quality=torch.stack(q)
        zero=sc.sum()*0.
        return (torch.stack(ctr).mean() if ctr else zero,
                torch.stack(giou).mean() if giou else zero,
                quality)

    def loss(self,batch,preds=None):
        if preds is not None:return super().loss(batch,preds=preds)
        if self.criterion is None:self.criterion=self.init_criterion()
        masks=self.build_alignment_valid_masks_from_batch(batch)
        out=self.predict(batch['rgb_img'],batch['tir_img'],rgb_semantic=batch['rgb_semantic_img'],
                         rgb_valid_mask=masks[0],tir_valid_mask=masks[1],
                         return_features=True,return_alignment_debug=True)
        native_batch={'img':batch['rgb_semantic_img'],'cls':batch['rgb_cls'],
                      'bboxes':batch['rgb_semantic_bboxes'],'batch_idx':batch['rgb_batch_idx']}
        raw,items=self.criterion(out['pred'],native_batch)
        losses=[raw.reshape(-1)]
        logs=self._loss_items_to_dict(items)
        B=int(batch['rgb_semantic_img'].shape[0]); p3=int(self.scale_to_index[3]); info=out['alignment_info']
        if self.alignment_loss_enabled:
            ali=self._alignment_auxiliary_loss(batch,info)
            weighted=self.lambda_alignment*ali['total']
            losses.append((weighted*B).reshape(1))
            logs['align_loss']=ali['total'].detach()
        rgb=out['rgb_backbone_raw'][p3];tir=out['tir_backbone'][p3];fused=out['fused_backbone'][p3]
        hw=tuple(rgb.shape[-2:])
        teachers=self.attach_frozen_teachers(rgb.device)
        micro=int(self.teacher_cfg.get('forward_microbatch',2))
        conf=float(self.teacher_cfg.get('min_conf',.15)); min_iou=float(self.teacher_cfg.get('min_gt_iou',.5))
        with torch.no_grad():
            rt=teachers.rgb.infer(batch['rgb_semantic_img'],batch['rgb_semantic_bboxes'],
                                  batch['rgb_batch_idx'],hw,microbatch=micro,
                                  conf_min=conf,correct_iou=min_iou)
            tt=teachers.tir.infer(batch['tir_img'],batch['tir_bboxes'],
                                  batch['tir_batch_idx'],hw,microbatch=micro,
                                  conf_min=conf,correct_iou=min_iou)
        rg=self._build_gaussian_target(batch['rgb_semantic_bboxes'],batch['rgb_batch_idx'],
            B,hw[0],hw[1],rgb.device,rgb.dtype,sigma_factor=.25,sigma_min_px=1.)
        tg=self._build_gaussian_target(batch['tir_bboxes'],batch['tir_batch_idx'],
            B,hw[0],hw[1],tir.device,tir.dtype,sigma_factor=.25,sigma_min_px=1.)
        pr=self.rgb_response(rgb);pt=self.tir_response(tir);pf=self.fused_response(fused)
        aux=(self._dense_response_loss(pr,rg)+self._dense_response_loss(pt,tg)+
             self._dense_response_loss(pf,rg))/3.
        native_rgb=self._masked_feature_loss(self.rgb_adapter(rgb),rt['p3'],rg,rt['score'])
        native_tir=self._masked_feature_loss(self.tir_adapter(tir),tt['p3'],tg,tt['score'])
        # RGB selective guard: only if a TP RGB teacher exists AND fused response
        # is worse than the auxiliary RGB response at foreground locations.
        with torch.no_grad():
            fg=rg.float()
            denom=fg.flatten(1).sum(1).clamp_min(1.)
            sr=(pr.float().sigmoid()*fg).flatten(1).sum(1)/denom
            sf=(pf.float().sigmoid()*fg).flatten(1).sum(1)/denom
            deficit=(sf+float(self.kcfg.get('rgb_margin',.03))<sr).float()
        guard_score=rt['score']*deficit
        rgb_guard=self._masked_feature_loss(self.fused_adapter(fused),rt['p3'],rg,guard_score)
        gc,gg,geo_quality=self._geometry(batch,info)
        rel_logit=out['dual_teacher_quality_logit'].flatten()
        rel_loss=F.binary_cross_entropy_with_logits(rel_logit.float(),geo_quality.detach())
        # Cross-coordinate TIR KD ONLY after geometry; no gradient through
        # pseudo-target or confidence. Geometry quality is TRAIN-GT-based,
        # the gate used at inference is the learned predicted reliability.
        with torch.no_grad():
            warped=self.alignment.warp_tensor(tt['response'],
                {'sampling_scale':info['sampling_scale'].detach(),
                 'sampling_translation':info['sampling_translation'].detach()},
                 output_hw=hw,padding_mode='zeros').clamp(0,1)
            valid=warped*(tt['score']*geo_quality).view(-1,1,1,1)
        dist=F.binary_cross_entropy_with_logits(pf.float(),warped.float(),reduction='none')
        tir_cross=(dist*valid).sum()/valid.sum().clamp_min(1.)
        k=self.kcfg
        terms={
          'aux_response':(aux,float(k.get('lambda_aux',.10))),
          'rgb_native':(native_rgb,float(k.get('lambda_rgb_native',.10))),
          'tir_native':(native_tir,float(k.get('lambda_tir_native',.10))),
          'rgb_guard':(rgb_guard,float(k.get('lambda_rgb_guard',.08))),
          'tir_cross':(tir_cross,float(k.get('lambda_tir_cross',.03))),
          'reliability':(rel_loss,float(k.get('lambda_reliability',.10))),
          'final_center':(gc,float(k.get('lambda_final_center',.02))),
          'final_giou':(gg,float(k.get('lambda_final_giou',.10))),
        }
        total=sum(v*w for v,w in terms.values())
        losses.append((total*B).reshape(1))
        for key,(val,weight) in terms.items():logs['dtkd_'+key]=val.detach()
        logs['dtkd_total']=total.detach()
        logs['dtkd_teacher_rgb_tp_rate']=(rt['score']>0).float().mean().detach()
        logs['dtkd_teacher_tir_tp_rate']=(tt['score']>0).float().mean().detach()
        logs['dtkd_rgb_guard_rate']=deficit.mean().detach()
        logs['dtkd_final_iou']=geo_quality.mean().detach()
        logs['dtkd_rel_pred']=out['dual_teacher_quality'].detach().mean()
        logs['dtkd_p3_alpha']=self.residual_alpha.detach().tanh()
        return torch.cat(losses),logs
