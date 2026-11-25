"""GMLS-v2 single shared YOLO Detect: training-only global auxiliary loss + K ROI.

ONE parameterized YOLO Detect module (inside self.detector) is reused during
training for global semantic and local ROI detection. During inference the
head is executed ONLY on predicted local ROI(s), never on global features.
No second independent YOLO detection head; no global/local box replacement.
"""
from __future__ import annotations
from copy import deepcopy
from contextlib import contextmanager
import torch
from torch import nn
import torch.nn.functional as F
from engine.evaluator import postprocess_predictions
from .model_base import (
    GMLSV2ROISingleHead, ROIGuide, _gt_centers, _guide_supervision,
    _extract_component_losses,
)
from .geometry import sample_roi_pair
from .multi_roi import labels_for_crops, per_image_merge
from .geo_losses import final_center_loss  # GEO_FINAL_CENTER_SCRATCH100_V2
from .top2_diverse import DiverseTop2Selector, SelectionOptions  # TOP2_DIVERSE_INTEGRATED_V1


class FeatureAwareGuide(ROIGuide):
    """Legacy RGB/TIR/J guidance + pretrained global P3 spatial features."""
    def __init__(self, feature_channels=512, hidden=32, temperature=.12):
        super().__init__(hidden=hidden,temperature=temperature)
        self.feature_branch=nn.Sequential(
            nn.Conv2d(feature_channels*2, hidden,1,bias=False),
            nn.BatchNorm2d(hidden),nn.SiLU(inplace=True),
            nn.Conv2d(hidden,1,1),
        )
        nn.init.normal_(self.feature_branch[-1].weight,std=.001)
        nn.init.zeros_(self.feature_branch[-1].bias)
        self.feature_strength=nn.Parameter(torch.tensor(.1))

    def forward(self,support,anchor,rgb_feature,tir_feature):
        logits,_,_=super().forward(support,anchor)
        feature_logits=self.feature_branch(torch.cat((rgb_feature,tir_feature),1))
        feature_logits=F.interpolate(feature_logits,size=logits.shape[-2:],mode='bilinear',align_corners=False)
        logits=logits+torch.tanh(self.feature_strength)*feature_logits
        B,_,H,W=logits.shape
        ys=torch.linspace(0,1,H,device=logits.device,dtype=torch.float32)
        xs=torch.linspace(0,1,W,device=logits.device,dtype=torch.float32)
        yy,xx=torch.meshgrid(ys,xs,indexing='ij')
        positions=torch.stack((xx,yy),-1).reshape(1,-1,2)
        p=F.softmax(logits.float().flatten(1)/max(self.temperature,.01),dim=1)
        center=(p.unsqueeze(-1)*positions).sum(1)
        return logits,p.reshape(B,1,H,W),center


def _second_candidate(logits,primary,anchor,temperature,min_separation=.10):
    """GT-free secondary ROI: GMLS anchor when separated, else another heatmap peak.

    Hard selection of the second peak only chooses a local neighborhood;
    weighted averaging within that neighborhood preserves gradients.
    """
    B,_,H,W=logits.shape
    ys=torch.linspace(0,1,H,device=logits.device,dtype=torch.float32)
    xs=torch.linspace(0,1,W,device=logits.device,dtype=torch.float32)
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    pos=torch.stack((xx,yy),dim=-1).reshape(1,H*W,2).expand(B,-1,-1)
    dist=(pos-primary.detach().float()[:,None,:]).norm(dim=-1)
    logits_flat=logits.float().flatten(1)
    masked=logits_flat.masked_fill(dist<min_separation,-1.e4)
    best_idx=masked.argmax(1)
    best_center=pos.gather(1,best_idx[:,None,None].expand(-1,1,2)).squeeze(1).detach()
    # Continuous within-peak refinement (local differentiable softargmax).
    sigma=.035
    local_prior=-((pos-best_center[:,None,:])**2).sum(-1)/(2*sigma*sigma)
    p=F.softmax(logits_flat/max(float(temperature),.01)+local_prior,dim=1)
    second=(p.unsqueeze(-1)*pos).sum(1)
    # Keep the true GMLS anchor as fallback when it offers a distinct ROI.
    distinct=(anchor.float()-primary.float()).norm(dim=-1)>=min_separation
    return torch.where(distinct[:,None],anchor.float(),second).clamp(0,1)


class GMLSV2OneHeadHybrid(GMLSV2ROISingleHead):
    architecture='gmls_v2_one_shared_head_global_aux_top2_officialpt_v2'

    def __init__(self,cfg,*,verbose=True):
        super().__init__(cfg,verbose=verbose)
        rc=cfg['model']['roi_single_head']
        self.num_rois=int(rc.get('topk',2))
        if self.num_rois not in (1,2):raise ValueError('topk must be 1 or 2')
        self.min_roi_separation=float(rc.get('min_roi_separation',.10))
        # TOP2_DIVERSE_INTEGRATED_V1: inference-only, optional per-YAML behavior.
        # In particular, Anti-UAV uses the original selection unless explicitly enabled.
        self.top2_diverse_enabled = bool(rc.get('top2_diverse_enabled', False))
        top2_opts = SelectionOptions(
            overlap_weight=float(rc.get('top2_diverse_overlap_weight', .20)),
            peak_weight=float(rc.get('top2_diverse_peak_weight', .04)),
            switch_margin=float(rc.get('top2_diverse_switch_margin', .01)),
            min_novel_mass=float(rc.get('top2_diverse_min_novel_mass', .02)),
            candidate_peaks=int(rc.get('top2_diverse_candidate_peaks', 24)),
        )
        self._top2_diverse = (
            DiverseTop2Selector(top2_opts, _second_candidate, self.roi_size)
            if self.top2_diverse_enabled and self.num_rois == 2 else None
        )

        self.guide=FeatureAwareGuide(
            feature_channels=int(rc.get('p3_channels',512)),
            hidden=int(rc.get('guide_hidden',32)),
            temperature=float(rc.get('temperature',.12)),
        )
        self.lambda_global=float(cfg['loss']['roi'].get('lambda_global_aux',.5))
        if not 0<=self.lambda_global<=2: raise ValueError('lambda_global_aux out of range')
        self.aux_bn_eval=bool(rc.get('aux_bn_eval',True))
        self.crop_nms_iou=float(cfg.get('eval',{}).get('crop_nms_iou',.5))
        self.max_det=int(cfg.get('eval',{}).get('max_det',300))
        self.gt_second_probability=float(cfg['train'].get('gt_second_prob_start',.80))
        self.gt_second_epochs=int(cfg['train'].get('gt_second_epochs',30))
        self.cfg=deepcopy(cfg)
        if verbose:
            print(f'[V2 HYBRID] train: shared one Detect runs on global & {self.num_rois} local ROI(s); '
                  f'val/test: ONLY {self.num_rois} local ROI(s), no global Detect',flush=True)
            print(f'[V2 HYBRID] global_aux_lambda={self.lambda_global} official_pt={self.official_path}',flush=True)

    def encode_and_crop(self,batch,*,return_debug=False,train_rois=False):
        g=self.global_encoder
        rgb=g.forward_rgb_backbone(batch['rgb_semantic_img'])
        tir=g.forward_tir_backbone(batch['tir_img'])
        mask_r,mask_t=g.build_alignment_valid_masks_from_batch(batch)
        aligned, info = g.alignment.align_pyramid(
            rgb_features=rgb,tir_features=tir,scale_to_index=g.scale_to_index,
            rgb_valid_mask=mask_r,tir_valid_mask=mask_t,return_debug=bool(return_debug))
        support=info['multi_target_guidance']
        anchor=info['coarse_anchor'].reshape(-1,2)
        p3_idx=g.scale_to_index[g.gtga_reference_scale]
        logits,guide_map,center=self.guide(support,anchor,rgb[p3_idx],aligned[p3_idx])
        primary=((1-self.anchor_mix)*center+self.anchor_mix*anchor).clamp(0,1)
        roi_centers=[primary]
        assisted=primary.new_zeros(())
        if self.num_rois==2:
            secondary=_second_candidate(logits,primary,anchor,self.guide.temperature,self.min_roi_separation)
            # TOP2_DIVERSE_INTEGRATED_V1: identical candidate scorer to the
            # previously validated inference ablation; never uses GT.
            # If legacy ablation has temporarily monkeypatched _second_candidate,
            # defer to that script so its baseline/diverse comparison stays valid.
            if (not train_rois and self._top2_diverse is not None and
                    _second_candidate is self._top2_diverse.base_fn):
                secondary = self._top2_diverse(
                    logits, primary, anchor, self.guide.temperature,
                    self.min_roi_separation, batch=batch, legacy_secondary=secondary,
                )

            if train_rois:
                gt,valid=_gt_centers(batch,primary)
                ratio=self.gt_second_probability*max(0.,1-self.epoch/max(self.gt_second_epochs,1))
                if ratio>0:
                    mask=(torch.rand(len(gt),device=gt.device)<ratio)&valid
                    jitter=float(self.cfg['train'].get('gt_jitter_high_px',48.))
                    rh=batch['rgb_ratio_pad'];rs=batch['rgb_semantic_ratio_pad']
                    factor=torch.stack([gt.new_tensor(float(s['ratio'])/float(h['ratio']))
                                        for h,s in zip(rh,rs)])
                    sem_h,sem_w=batch['rgb_semantic_img'].shape[-2:]
                    offsets=(torch.rand_like(gt)*2-1)*jitter*factor[:,None]/gt.new_tensor([max(sem_w-1,1),max(sem_h-1,1)])
                    secondary=torch.where(mask[:,None],(gt+offsets).clamp(0,1),secondary)
                    assisted=mask.float().mean()
            roi_centers.append(secondary)
        B=len(primary);K=len(roi_centers)
        chosen=torch.stack(roi_centers,dim=1).reshape(B*K,2)
        def repeat(x):return x.repeat_interleave(K,dim=0)
        high_metas=[m for m in batch['rgb_ratio_pad'] for _ in range(K)]
        sem_metas=[m for m in batch['rgb_semantic_ratio_pad'] for _ in range(K)]
        crops_rgb,crops_tir,geometry=sample_roi_pair(
            repeat(batch['rgb_img']),repeat(batch['tir_img']),chosen,
            repeat(info['sampling_scale']),repeat(info['sampling_translation']),
            high_metas,sem_metas,batch['rgb_semantic_img'].shape[-2:],size=self.roi_size,
            center_grad_scale=self.center_grad_scale if train_rois else 0.,
            geometry_grad_scale=self.geometry_grad_scale if train_rois else 0.,
            tir_size=getattr(self,'local_tir_roi_size',None),  # CDGR_SMF_LOCAL_TIR_SIZE_V1
        )
        return {'rgb_roi':crops_rgb,'tir_roi':crops_tir,'roi':geometry,
                'guide_logits':logits,'guide_map':guide_map,'center_pred':primary,
                'center_used':chosen,'gt_assisted':assisted,
                'alignment_info':info,'num_rois':B*K,'rois_per_image':K,
                'global_rgb_features':rgb if train_rois else None,
                'global_tir_aligned':aligned if train_rois else None}

    @contextmanager
    def _global_head_bn_policy(self):
        """Do not contaminate local Detect running BN statistics with global features.

        BN affine parameters remain trainable, and global detection gradients
        still flow. Only BN running mean/var updates are disabled for the
        global auxiliary pass (not the local pass).
        """
        if not self.aux_bn_eval:
            yield
            return
        modules=[]
        for layer in list(self.detector.model)[self.detector.backbone_len:]:
            for m in layer.modules():
                if isinstance(m,nn.modules.batchnorm._BatchNorm):
                    modules.append((m,m.training))
                    m.eval()
        try:
            yield
        finally:
            for m,was_training in modules:m.train(was_training)

    def _shared_head_global_prediction(self,ex):
        """Training-only supervision; exactly the SAME YOLO head as local detector."""
        rgb=ex['global_rgb_features'];aligned=ex['global_tir_aligned']
        g=self.global_encoder
        fused=list(rgb)
        for idx in g.fusion_indices:
            fused[idx]=rgb[idx]+aligned[idx]
        with self._global_head_bn_policy():
            return self.detector.forward_head(fused)

    def loss(self,batch):
        ex=self.encode_and_crop(batch,return_debug=True,train_rois=True)
        B=int(batch['rgb_img'].shape[0]);K=ex['rois_per_image']
        labels,npositive=labels_for_crops(batch,ex['roi'],K,self.min_visible)
        # EMPTYGT_ASYMMETRIC_DETECTION_MASK_V1
        # Native Detect is supervised in RGB coordinates. When TIR has GT
        # but RGB lacks it, this image is annotation-unknown, NOT a negative.
        # RGB-only and both-empty pairs remain eligible for RGB detection.
        mask_asymmetric=bool(self.cfg.get('data',{}).get('ignore_rgb_empty_tir_positive',False))
        if mask_asymmetric:
            from .empty_gt_adaptation import (
                rgb_supervision_mask, select_rgb_local_labels,
                select_global_training_data,
            )
            rgb_supervised=rgb_supervision_mask(batch,B)
            local_labels=select_rgb_local_labels(
                labels,ex['rgb_roi'],ex['tir_roi'],rgb_supervised,K)
        else:
            rgb_supervised=torch.ones(B,device=batch['rgb_img'].device,dtype=torch.bool)
            local_labels=dict(labels)
            local_labels['rgb_img']=ex['rgb_roi']
            local_labels['tir_img']=ex['tir_roi']
        if local_labels is not None:
            # Det criterion lazily initialized by the native v5 RGB-T detector.
            local_raw,local_items=self.detector(local_labels)
            local_loss=local_raw.sum()*(B/float(B*K))
        else:
            # All RGB labels missing while TIR has GT. Do NOT train these as
            # background. A graph-connected zero keeps DDP forward valid.
            local_loss=(ex['rgb_roi'].float().sum()+ex['tir_roi'].float().sum())*0.0
            local_items={}
        # Independent global Neck+Detect; same RGB-labelled image subset.
        if self.lambda_global > 0 and bool(rgb_supervised.any()):
            if mask_asymmetric:
                global_ex,global_target=select_global_training_data(ex,batch,rgb_supervised)
            else:
                global_ex=ex
                global_target={'img':batch['rgb_semantic_img'], 'cls':batch['rgb_cls'],
                               'bboxes':batch['rgb_semantic_bboxes'],'batch_idx':batch['rgb_batch_idx']}
            global_pred=self._shared_head_global_prediction(global_ex)
            global_raw,global_items=self.detector.criterion(global_pred,global_target)
            global_loss=global_raw.sum()
        else:
            global_loss=local_loss.new_zeros(())
            global_items={}
        g=self.global_encoder
        if g.alignment_loss_enabled:
            stats=g._alignment_auxiliary_loss(batch,ex['alignment_info'])
            alignment_loss=stats['total']*g.lambda_alignment*B
        else:
            stats={}
            alignment_loss=local_loss.new_zeros(())
        # GEO_FINAL_CENTER_SCRATCH100_V2: supervise the ACTUAL final Warp,
        # q=S*p+t. Only 1:1 RGB/TIR GT during training; never inference.
        geo_cfg=self.cfg.get('loss',{}).get('alignment',{})
        final_center_weight=float(geo_cfg.get('lambda_final_center',0.0))
        if final_center_weight>0.0 and g.alignment_loss_enabled:
            final_center_raw,final_center_pairs,final_center_error=final_center_loss(
                batch,ex['alignment_info'],
                pixel_norm=float(geo_cfg.get('final_center_norm_px',8.0)),
                beta=float(geo_cfg.get('final_center_beta',0.5)))
            geo_warm=max(0,int(geo_cfg.get('final_center_warmup_epochs',10)))
            geo_ramp=min(1.,(float(getattr(self,'epoch',0))+1.)/geo_warm) if geo_warm else 1.
            alignment_loss=alignment_loss+final_center_weight*geo_ramp*B*final_center_raw
        else:
            final_center_raw=alignment_loss.new_zeros(())
            final_center_pairs=0
            final_center_error=alignment_loss.new_zeros(())
            geo_ramp=0.
        gt,valid=_gt_centers(batch,ex['center_pred'])
        guide_loss,center_loss=_guide_supervision(
            ex['guide_logits'],ex['center_pred'],gt,valid,
            sigma_px=float(self.cfg['loss']['roi'].get('sigma_px',2.5)),
            temperature=self.guide.temperature)
        total=(self.lambda_local*local_loss+self.lambda_global*global_loss+
               alignment_loss+B*self.lambda_guide*guide_loss+B*self.lambda_center*center_loss)
        parts={'total':total.detach(),'local_det':local_loss.detach(),
               'global_aux_det':global_loss.detach(),'global_aux_weighted':(global_loss*self.lambda_global).detach(),
               'align_weighted':alignment_loss.detach(),
               'roi_guide':guide_loss.detach(),'roi_center':center_loss.detach(),
               'roi_positive':float(npositive)/max(B*K,1),
               'roi_tir_valid':ex['roi'].tir_valid.mean().detach(),
               'roi_count':float(K), 'gt_assist':ex['gt_assisted'].detach(),
               'emptygt_rgb_supervised':float(rgb_supervised.sum()),
               'emptygt_rgb_unknown_ignored':float((~rgb_supervised).sum())}
        parts['align_final_center']=final_center_raw.detach()
        parts['align_final_center_px']=final_center_error.detach()
        parts['align_final_center_pairs']=float(final_center_pairs)
        parts['align_final_center_ramp']=float(geo_ramp)
        for prefix,raw_items in [('local',local_items),('global_aux',global_items)]:
            components=_extract_component_losses(raw_items,prefix)
            parts.update(components)
            # Existing Ultralytics heads often return box_loss/cls_loss/dfl_loss
            # whereas the prior logger only printed the alias local_box etc.
            for name in ('box','cls','dfl'):
                for candidate in (f'{prefix}_{name}',f'{prefix}_{name}_loss',
                                  f'{prefix}_{name}Loss'):
                    if candidate in components:
                        parts[f'{prefix}_{name}']=components[candidate]
                        break
        for name,key in [('anchor','anchor'),('coarse','coarse'),('scale','scale'),
                         ('targetness','targetness'),('v2_quality','v2_quality'),
                         ('v2_route','v2_route'),('gmls_gt_quality','v2_gmls_gt_quality'),
                         ('dmls_gt_quality','v2_dmls_gt_quality')]:
            if key in stats:parts['align_'+name]=stats[key].detach()
        return total,parts

    @torch.no_grad()
    def predict(self,batch,*,conf=.001,iou=.70):
        if self.training:raise RuntimeError('Call model.eval() first')
        ex=self.encode_and_crop(batch,return_debug=False,train_rois=False)
        raw=self.detector.predict(ex['rgb_roi'],ex['tir_roi'])
        loc=postprocess_predictions(raw,self.detector,conf_thres=conf,iou_thres=iou)
        preds=per_image_merge(loc,ex['roi'],batch,ex['rois_per_image'],
                              conf=conf,nms_iou=self.crop_nms_iou,max_det=self.max_det)
        return preds,ex
