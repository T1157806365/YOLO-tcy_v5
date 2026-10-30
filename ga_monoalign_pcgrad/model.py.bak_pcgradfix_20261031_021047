"""Original trainable GMLS-v2 student plus two FULL single-modal YOLO neck/detection heads.

The core forward graph is reused verbatim. RGB/TIR auxiliary detection and teacher
attention backpropagate to their corresponding backbone. The GMLS-v2 estimator is
never replaced or detached. Feature-gradient PCGrad only changes the *auxiliary*
gradient component when it opposes the alignment loss at P3/P4/P5 outputs.
"""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from models.rgbt_model import get_layer_input
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from .teachers import DualTeachers


class NativeDetectionHead(nn.Module):
    """Original YOLO26m neck + full Detect head, cloned for one modality."""
    def __init__(self, core):
        super().__init__()
        self.backbone_len=int(core.backbone_len)
        # Module indices and .f references are kept identical to the source.
        self.model=nn.ModuleList([deepcopy(layer) for layer in list(core.model)[self.backbone_len:]])
        self.nc=int(core.nc)
        self.names=core.names
        self.stride=core.stride
        self.args=deepcopy(core.args)
        self.end2end=bool(getattr(core,'end2end',False))
        self.criterion=None

    def forward(self, backbone_outputs):
        outputs=list(backbone_outputs)
        if len(outputs)!=self.backbone_len:
            raise RuntimeError('Backbone layer count mismatch in auxiliary head')
        x=outputs[-1]
        for layer in self.model:
            x=layer(get_layer_input(layer,x,outputs))
            outputs.append(x)
        return x

    def calc_loss(self,pred,batch):
        if self.criterion is None:
            self.criterion=E2ELoss(self) if self.end2end else v8DetectionLoss(self)
        return self.criterion(pred,batch)


class AlignmentProtectedStudent(nn.Module):
    def __init__(self,core,cfg):
        super().__init__()
        self.core=core
        self.aux_rgb=NativeDetectionHead(core)
        self.aux_tir=NativeDetectionHead(core)
        self.cfg=cfg
        self.model_name=core.model_name
        self.fusion_type=core.fusion_type
        self.nc=core.nc;self.names=core.names;self.end2end=core.end2end
        self.stage='stage1';self.epoch=1
        # Teachers are only in-memory training utilities; exclude them from
        # student parameters, optimizer and saved checkpoint.
        object.__setattr__(self,'_teachers',None)

    def set_progress(self,stage,epoch):
        self.stage=str(stage);self.epoch=int(epoch)

    def attach_teachers(self,device):
        obj=self._teachers
        if obj is None:
            obj=DualTeachers(self.cfg['dual_teachers'],int(self.core.scale_to_index[3]),device)
            object.__setattr__(self,'_teachers',obj)
        return obj

    def build_alignment_valid_masks_from_batch(self,batch):
        return self.core.build_alignment_valid_masks_from_batch(batch)

    def forward(self,rgb,tir=None,rgb_semantic=None,**kwargs):
        if isinstance(rgb,dict):return self.loss(rgb)
        # Inference is EXACT original model: auxiliary heads do not participate.
        return self.core.predict(rgb,tir,rgb_semantic=rgb_semantic,**kwargs)

    def predict(self,rgb,tir,rgb_semantic=None,**kwargs):
        return self.core.predict(rgb,tir,rgb_semantic=rgb_semantic,**kwargs)

    def aux_predict(self,modality,rgb_img=None,tir_img=None):
        if modality=='rgb':
            return self.aux_rgb(self.core.forward_rgb_backbone(rgb_img))
        if modality=='tir':
            return self.aux_tir(self.core.forward_tir_backbone(tir_img))
        raise ValueError(modality)

    def schedule(self):
        """Piecewise weights follow requested 300+200 training protocol."""
        if self.stage=='stage1':
            e=self.epoch
            if e<=60: return 0.,0.
            if e<=100:return 0.10,0.05*(e-60)/40.
            if e<=250:return 0.15,0.05
            return 0.10,0.03
        e=self.epoch
        if e<=100:return 0.08,0.02
        return 0.05,0.02*max(0.,(200-e)/100.)

    @staticmethod
    def _attention(x):
        z=x.float().abs().mean(dim=1)
        mean=z.mean((-2,-1),keepdim=True)
        std=z.std((-2,-1),unbiased=False,keepdim=True).clamp_min(1e-4)
        return (z-mean)/std

    @staticmethod
    def _roi_mask(boxes,idx,bsz,h,w,device):
        """ROI using native GT; padded at least 3 P3 cells for tiny UAVs."""
        mask=torch.zeros(bsz,h,w,device=device,dtype=torch.bool)
        for box,b in zip(boxes.detach().float().cpu(),idx.detach().long().cpu()):
            i=int(b);cx,cy,bw,bh=[float(v) for v in box]
            x1=max(0,int((cx-bw/2)*w)-3);x2=min(w,int((cx+bw/2)*w)+4)
            y1=max(0,int((cy-bh/2)*h)-3);y2=min(h,int((cy+bh/2)*h)+4)
            mask[i,y1:y2,x1:x2]=True
        return mask

    def _kd_one(self,student,teacher,good,boxes,idx):
        b,_,h,w=student.shape
        roi=self._roi_mask(boxes,idx,b,h,w,student.device)
        stu=self._attention(student)
        tea=self._attention(teacher['p3']).detach()
        losses=[]
        for i in range(b):
            if not bool(good[i]) or not bool(roi[i].any()):continue
            s=stu[i][roi[i]];t=tea[i][roi[i]]
            if s.numel()<2:continue
            # Scale-invariant, spatial soft attention on the UAV neighborhood.
            losses.append(F.kl_div(F.log_softmax(s,0),F.softmax(t,0),reduction='sum'))
        if not losses:return stu.sum()*0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _native(batch,modal):
        if modal=='rgb':
            return {'img':batch['rgb_semantic_img'],'cls':batch['rgb_cls'],
                    'bboxes':batch['rgb_semantic_bboxes'],'batch_idx':batch['rgb_batch_idx']}
        return {'img':batch['tir_img'],'cls':batch['tir_cls'],
                'bboxes':batch['tir_bboxes'],'batch_idx':batch['tir_batch_idx']}

    @staticmethod
    def _pcgrad_feature_hooks(align_loss, auxiliary_loss, rgb_feats, tir_feats, enabled=True):
        """Conflict-protect auxiliary gradient at feature *outputs*, not parameters.

        Uses autograd.grad only on intermediary features: DDP parameter reducers
        are not entered before standard backward. On negative cosine, project
        the auxiliary feature gradient to be orthogonal to alignment gradient.
        All ranks apply their local feature surgery, DDP then allreduces the
        resulting parameter gradients in the ordinary backward call.
        """
        diag={}
        if not enabled or not align_loss.requires_grad or not auxiliary_loss.requires_grad:
            return diag
        pending=[]
        for name,fs in (('rgb',rgb_feats),('tir',tir_feats)):
            feats=[x for x in fs if torch.is_tensor(x) and x.requires_grad]
            if not feats:continue
            ga=torch.autograd.grad(align_loss,feats,retain_graph=True,allow_unused=True)
            gm=torch.autograd.grad(auxiliary_loss,feats,retain_graph=True,allow_unused=True)
            dot=align_loss.new_zeros(());na=dot.clone();nm=dot.clone()
            for a,m in zip(ga,gm):
                if a is None or m is None:continue
                a=a.detach().float();m=m.detach().float()
                dot=dot+(a*m).sum();na=na+a.square().sum();nm=nm+m.square().sum()
            cosine=dot/(na.sqrt()*nm.sqrt()+1e-12)
            conflict=bool((dot<0).item() and (na>1e-20).item())
            diag[f'pcgrad_{name}_cosine']=cosine.detach()
            diag[f'pcgrad_{name}_conflict']=float(conflict)
            if conflict:
                coef=(dot/na.clamp_min(1e-12)).detach()
                for feat,a,m in zip(feats,ga,gm):
                    if a is None or m is None:continue
                    pending.append((feat,(-coef*a.detach()).to(feat.dtype)))
        # Install hooks only AFTER all gradient probes: otherwise the RGB
        # correction could contaminate the TIR probe on a shared graph.
        for feat,correction in pending:
            feat.register_hook(lambda g,delta=correction:g+delta)
        return diag

    def loss(self,batch):
        core=self.core
        if core.criterion is None:core.criterion=core.init_criterion()
        bsz=batch['rgb_semantic_img'].shape[0]
        masks=core.build_alignment_valid_masks_from_batch(batch)
        features=core.forward_features(batch['rgb_img'],batch['tir_img'],
             rgb_semantic=batch['rgb_semantic_img'],rgb_valid_mask=masks[0],
             tir_valid_mask=masks[1],return_alignment_debug=True)
        main_pred=core.forward_head(features['fused_backbone'])
        det_vec,det_items=core.criterion(main_pred,self._native(batch,'rgb'))
        det=det_vec.sum()
        align=det*0.
        logs=core._loss_items_to_dict(det_items)
        if core.alignment_loss_enabled:
            astat=core._alignment_auxiliary_loss(batch,features['alignment_info'])
            align=float(core.lambda_alignment)*astat['total']*bsz
            logs.update({f'align_{k}':v.detach() if torch.is_tensor(v) else float(v)
                         for k,v in astat.items()})
        # Both auxiliary heads are run even when warmup weight=0; zero-multiplied
        # losses keep all parameters participating in DDP's training graph.
        rpred=self.aux_rgb(features['rgb_backbone_raw'])
        tpred=self.aux_tir(features['tir_backbone'])
        lr,lr_items=self.aux_rgb.calc_loss(rpred,self._native(batch,'rgb'))
        lt,lt_items=self.aux_tir.calc_loss(tpred,self._native(batch,'tir'))
        wm,wk=self.schedule()
        mono=wm*(lr.sum()+lt.sum())
        # KD only after warmup, its teachers never participate in inference.
        kd=det*0.
        kr=kd;kt=kd; rgb_rate=0.;tir_rate=0.
        if wk>0:
            teachers=self._teachers
            if teachers is None:
                raise RuntimeError('Training with KD requires student.attach_teachers(device)')
            with torch.no_grad():
                tr,tt=teachers.get(batch,features['rgb_backbone_raw'][core.scale_to_index[3]].shape[-2:])
            rid=core.scale_to_index[3]
            kr=self._kd_one(features['rgb_backbone_raw'][rid],tr,tr['good'],
                            batch['rgb_semantic_bboxes'],batch['rgb_batch_idx'])
            kt=self._kd_one(features['tir_backbone'][rid],tt,tt['good'],
                            batch['tir_bboxes'],batch['tir_batch_idx'])
            rgb_rate=float(tr['good'].float().mean().item())
            tir_rate=float(tt['good'].float().mean().item())
            kd=wk*bsz*(kr+kt)
        # Geometry gradients vs mono+KD at the SAME backbone output interfaces.
        indices=[core.scale_to_index[s] for s in (3,4,5)]
        rg=[features['rgb_backbone_raw'][i] for i in indices]
        tg=[features['tir_backbone'][i] for i in indices]
        diags=self._pcgrad_feature_hooks(align,mono+kd,rg,tg,
                    enabled=bool(self.cfg['monoalign'].get('pcgrad_enabled',True)) and wm>0)
        # Scalar losses packed in a vector for existing trainer-style logs.
        total=torch.stack((det,align,mono,kd))
        logs.update({'rgb_aux_loss':lr.sum().detach()/bsz,
                     'tir_aux_loss':lt.sum().detach()/bsz,
                     'det_raw_loss':det.detach()/bsz,
                     'align_weighted_loss':align.detach()/bsz,
                     'mono_weighted_loss':mono.detach()/bsz,
                     'kd_weighted_loss':kd.detach()/bsz,
                     'rgb_kd_response':kr.detach(),
                     'tir_kd_response':kt.detach(),
                     'teacher_rgb_selected':rgb_rate,
                     'teacher_tir_selected':tir_rate,
                     'weight_mono':wm,'weight_kd':wk,
                     **diags})
        return total, logs
