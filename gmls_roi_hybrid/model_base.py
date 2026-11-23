"""GMLS-v2 guides native HR RGB-T ROI; ONE shared YOLO26m detection head.

Global path: official-YOLO-initialized RGB/TIR backbones + original RLSFA/GTGA
             including GMLS-v2/DMLS, NO global YOLO detection head.
Guide path:  RGB/TIR/aligned joint target maps + coarse anchor -> 2D ROI heatmap.
Local path:  differentiable 640x640 RGB high crop + GTGA-aligned TIR crop ->
             official-YOLO-initialized dual backbone + ADD + ONE YOLO Detect.

GT-assisted cropping is TRAIN-ONLY and smoothly annealed away. At eval neither
ROI center nor geometry is supplied from GT labels. All gradients are limited
by configurable ROI center/geometry gradient scales.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from models.gtga_bhlr_factory import build_model_from_config
from models.rgbt_model import RGBTDetectionModel
from utils.config import resolve_model
from .geometry import sample_roi_pair, local_labels_from_high, boxes_roi_to_semantic, roi_gt_coverage


class ROIGuide(nn.Module):
    def __init__(self, hidden: int = 32, temperature: float = 0.12):
        super().__init__()
        self.temperature=float(temperature)
        self.conv=nn.Sequential(
            nn.Conv2d(4,hidden,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden),nn.SiLU(inplace=True),
            nn.Conv2d(hidden,hidden,3,padding=1,bias=False),
            nn.BatchNorm2d(hidden),nn.SiLU(inplace=True),
            nn.Conv2d(hidden,1,1,bias=True),
        )
        nn.init.normal_(self.conv[-1].weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, support: torch.Tensor, anchor: torch.Tensor):
        if support.shape[1]!=3 or anchor.ndim!=2 or anchor.shape[1]!=2:
            raise ValueError(f'Invalid ROI guide inputs {support.shape}, {anchor.shape}')
        b,_,h,w=support.shape
        # Detector-local feature maps are sampled with align_corners=True.
        ys=torch.linspace(0,1,h,device=support.device,dtype=torch.float32)
        xs=torch.linspace(0,1,w,device=support.device,dtype=torch.float32)
        yy,xx=torch.meshgrid(ys,xs,indexing='ij')
        pos=torch.stack((xx,yy),-1).unsqueeze(0)
        prior=torch.exp(-0.5*(((pos-anchor.float()[:,None,None,:])/0.08)**2).sum(-1))
        inp=torch.cat((support.float(),prior[:,None]),1)
        logits=self.conv(inp)
        probs=F.softmax(logits.float().flatten(1)/max(self.temperature,0.01),dim=1)
        normalized_center=(probs.unsqueeze(-1)*pos.reshape(1,-1,2)).sum(1)
        return logits,probs.reshape(b,1,h,w),normalized_center


def _gt_centers(batch: dict, reference: torch.Tensor):
    """Primary UAV per-image training center in normalized RGB-semantic coords.

    If several targets exist, the largest GT is used to supervise the single
    ROI; all objects inside the ROI still receive detection supervision.
    """
    b=reference.shape[0]
    out=reference.detach().clone();valid=torch.zeros(b,device=reference.device,dtype=torch.bool)
    idx=batch['rgb_batch_idx'].long()
    for i in range(b):
        bb=batch['rgb_semantic_bboxes'][idx==i]
        if len(bb):
            k=bb[:,2:].prod(-1).argmax()
            out[i]=bb[k,:2]
            valid[i]=True
    return out,valid


def _guide_supervision(logits:torch.Tensor, pred:torch.Tensor, gt:torch.Tensor,
                       valid:torch.Tensor, sigma_px:float=2.5, temperature:float=0.12):
    b,_,h,w=logits.shape
    ys=torch.linspace(0,1,h,device=logits.device,dtype=torch.float32)
    xs=torch.linspace(0,1,w,device=logits.device,dtype=torch.float32)
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    xy=torch.stack((xx,yy),-1)[None]
    sigma=gt.new_tensor([sigma_px/max(w-1,1),sigma_px/max(h-1,1)])
    density=torch.exp(-0.5*((xy-gt[:,None,None,:])/sigma).square().sum(-1))
    target=density.flatten(1)/density.flatten(1).sum(1,keepdim=True).clamp_min(1e-8)
    logp=F.log_softmax(logits.float().flatten(1)/max(temperature,0.01),dim=1)
    validf=valid.float()
    crossentropy=(-target*logp).sum(1)
    guide=(crossentropy*validf).sum()/validf.sum().clamp_min(1)
    # 640 semantic pixel error rather than only a tiny normalized loss.
    # EMPTYGT_ASYMMETRIC_CENTER_GRAD_FIX_V1
    # Missing RGB GT sets gt=pred.detach(), giving exactly zero distance.
    # sqrt(0) has singular backward: even 0*invalid_mask produces NaN.
    # Index only valid samples and use a zero-safe vector norm.
    if bool(valid.any()):
        center=torch.linalg.vector_norm((pred[valid]-gt[valid]).float(),dim=-1).mean()
    else:
        center=pred.float().sum()*0.0
        guide=logits.float().sum()*0.0
    return guide,center


def _extract_component_losses(items, prefix):
    if isinstance(items,dict):
        return {prefix+'_'+k: v.detach() if torch.is_tensor(v) else v for k,v in items.items()}
    if torch.is_tensor(items):
        vals=items.detach().reshape(-1)
    elif isinstance(items,(tuple,list)):
        vals=torch.stack([x.detach() if torch.is_tensor(x) else torch.tensor(float(x)) for x in items]).reshape(-1)
    else:
        return {prefix+'_loss':float(items)}
    if len(vals)==3:
        return {prefix+'_box':vals[0],prefix+'_cls':vals[1],prefix+'_dfl':vals[2]}
    return {prefix+f'_loss_{i}':v for i,v in enumerate(vals)}


class GMLSV2ROISingleHead(nn.Module):
    architecture='gmls_v2_roi_single_head_officialpt_v1'

    def __init__(self,cfg:dict,*,verbose:bool=True):
        super().__init__()
        self.cfg=deepcopy(cfg)
        mc=cfg['model']; rc=mc['roi_single_head'];tc=cfg['train']
        if not bool(rc.get('enabled',False)):
            raise ValueError('Set model.roi_single_head.enabled: true')
        if str(mc.get('name',''))!='yolo26m':
            raise ValueError('This release requires model.name: yolo26m')
        if mc.get('variant')!='rlsfa_gtga_centered_bhlr_v2':
            raise ValueError('Requires original GMLS-v2 variant')
        if mc.get('pretrained') is not True or rc.get('local_pretrained') is not True:
            raise ValueError('Both model.pretrained and roi_single_head.local_pretrained must be TRUE')
        if bool(mc.get('bhlr',{}).get('enabled',True)):
            raise ValueError('Single-head native ROI replaces BHLR; set model.bhlr.enabled: false')
        if bool(mc.get('hvad',{}).get('enabled',False)) or bool(mc.get('tri_gate',{}).get('enabled',False)):
            raise ValueError('No HVAD or TriGate in this variant')
        if int(rc.get('roi_size',640))!=640:
            raise ValueError('Current release supports 640x640 local ROIs')
        self.official_path=Path(resolve_model('yolo26m',pretrained=True)['weights'])
        if not self.official_path.is_file():
            raise FileNotFoundError(f'Missing official YOLO26m weights: {self.official_path}')
        # Build original GMLS-v2 (including trained GTGA/GMLS submodules), then
        # PRUNE global neck/head. Consequently there is one Detect head in net.
        global_cfg=deepcopy(cfg)
        global_cfg['model']['bhlr']['enabled']=False
        self.global_encoder=build_model_from_config(global_cfg,pretrained=True,verbose=verbose)
        self.global_encoder.model=nn.ModuleList(
            list(self.global_encoder.model)[:self.global_encoder.backbone_len])
        if hasattr(self.global_encoder,'fusions'):
            del self.global_encoder.fusions  # never used by encoder-only stage
        self.global_encoder.criterion=None
        # Independent RGB/TIR local YOLO26m backbone, ADD and *single* Detect.
        self.detector=RGBTDetectionModel(
            model_name='yolo26m', nc=1, pretrained=True, fusion='add',
            fusion_indices=[4,6,10], align_mode='bilinear',
            names={0:'UAV'}, verbose=verbose)
        self.detector.args.box=float(cfg.get('loss',{}).get('box',7.5))
        self.detector.args.cls=float(cfg.get('loss',{}).get('cls',0.5))
        self.detector.args.dfl=float(cfg.get('loss',{}).get('dfl',1.5))
        self.detector.criterion=None
        # Enforce exactly ONE YOLO bounding-box detection head in the model.
        # GMLS-v2 targetness/quality heads are localization auxiliaries, not Detect.
        detect_heads=[module for module in self.modules()
                      if hasattr(module,'nl') and hasattr(module,'nc')
                      and 'detect' in type(module).__name__.lower()]
        if len(detect_heads)!=1:
            raise RuntimeError(f'Expected exactly 1 YOLO Detect, got {len(detect_heads)}')
        self.guide=ROIGuide(int(rc.get('guide_hidden',32)),float(rc.get('temperature',.12)))
        self.anchor_mix=float(rc.get('anchor_mix',.30))
        self.roi_size=int(rc.get('roi_size',640))
        self.lambda_guide=float(cfg['loss']['roi']['lambda_guide'])
        self.lambda_center=float(cfg['loss']['roi']['lambda_center'])
        self.lambda_local=float(cfg['loss']['roi']['lambda_local'])
        self.center_grad_scale=float(rc.get('center_grad_scale',.1))
        self.geometry_grad_scale=float(rc.get('geometry_grad_scale',.1))
        self.min_visible=float(rc.get('min_gt_visible',.60))
        self.epoch=0
        self.nc=1
        self.names={0:'UAV'}
        if verbose:
            print(f'[OFFICIAL YOLO PT] {self.official_path} loaded into global RGB/TIR and local RGB/TIR',flush=True)
            print('[ARCHITECTURE] GMLS-v2 global encoder + ROI guidance + native RGB/TIR crop + ONE YOLO26m Detect',flush=True)

    def set_epoch(self,epoch:int):
        self.epoch=int(epoch)

    def encode_and_crop(self,batch,*,return_debug=False,train_rois=False):
        g=self.global_encoder
        rgb_feat=g.forward_rgb_backbone(batch['rgb_semantic_img'])
        tir_feat=g.forward_tir_backbone(batch['tir_img'])
        mask_r,mask_t=g.build_alignment_valid_masks_from_batch(batch)
        _,info=g.alignment.align_pyramid(
            rgb_features=rgb_feat,tir_features=tir_feat,
            scale_to_index=g.scale_to_index,rgb_valid_mask=mask_r,
            tir_valid_mask=mask_t,return_debug=bool(return_debug))
        support=info['multi_target_guidance']
        anchor=info['coarse_anchor'].reshape(-1,2)
        logits,guide_map,center=self.guide(support,anchor)
        pred=(1-self.anchor_mix)*center + self.anchor_mix*anchor
        chosen=pred
        assisted=pred.new_zeros(())
        if train_rois:
            gt,valid=_gt_centers(batch,pred)
            tcfg=self.cfg['train']
            fade=max(1,int(tcfg.get('gt_assist_epochs',20)))
            ratio=float(tcfg.get('gt_assist_start',.8))*max(0.0,1.0-self.epoch/fade)
            if ratio>0:
                assisted_mask=(torch.rand(len(pred),device=pred.device)<ratio)&valid
                # Jitter in HIGH pixels mapped to semantic normalized coordinates.
                hm=batch['rgb_ratio_pad'];sm=batch['rgb_semantic_ratio_pad']
                _,_,hh,hw=batch['rgb_img'].shape
                _,_,sh,sw=batch['rgb_semantic_img'].shape
                rel=torch.stack([gt.new_tensor(float(s['ratio'])/float(h['ratio']))
                                 for h,s in zip(hm,sm)])
                jitter=float(tcfg.get('gt_jitter_high_px',48.))
                offsets=(torch.rand_like(gt)*2-1)*jitter * rel[:,None] / gt.new_tensor([max(sw-1,1),max(sh-1,1)])
                chosen=torch.where(assisted_mask[:,None],(gt+offsets).clamp(0,1),pred)
                assisted=assisted_mask.float().mean()
        crop_rgb,crop_tir,roi=sample_roi_pair(
            batch['rgb_img'],batch['tir_img'],chosen,
            info['sampling_scale'],info['sampling_translation'],
            batch['rgb_ratio_pad'],batch['rgb_semantic_ratio_pad'],
            batch['rgb_semantic_img'].shape[-2:],size=self.roi_size,
            geometry_grad_scale=self.geometry_grad_scale if train_rois else 0,
            center_grad_scale=self.center_grad_scale if train_rois else 0,
        )
        return {
            'rgb_roi':crop_rgb,'tir_roi':crop_tir,'roi':roi,
            'guide_logits':logits,'guide_map':guide_map,
            'center_pred':pred,'center_used':chosen,
            'gt_assisted':assisted,'alignment_info':info,
        }

    def forward(self,batch:dict):
        return self.loss(batch)

    def loss(self,batch):
        extracted=self.encode_and_crop(batch,return_debug=True,train_rois=True)
        roi=extracted['roi']
        local,n_pos=local_labels_from_high(batch,roi,self.min_visible)
        local['rgb_img']=extracted['rgb_roi']
        local['tir_img']=extracted['tir_roi']
        raw,items=self.detector(local)
        detector_loss=raw.sum()
        g=self.global_encoder
        if g.alignment_loss_enabled:
            align_stats=g._alignment_auxiliary_loss(batch,extracted['alignment_info'])
            align_loss=align_stats['total']*g.lambda_alignment*int(batch['rgb_img'].shape[0])
        else:
            align_stats={}
            align_loss=detector_loss.new_zeros(())
        gt,valid=_gt_centers(batch,extracted['center_pred'])
        guide_loss,center_loss=_guide_supervision(
            extracted['guide_logits'],extracted['center_pred'],gt,valid,
            sigma_px=float(self.cfg['loss']['roi'].get('sigma_px',2.5)),
            temperature=self.guide.temperature)
        b=int(batch['rgb_img'].shape[0])
        total=(self.lambda_local*detector_loss+align_loss+
               b*self.lambda_guide*guide_loss+b*self.lambda_center*center_loss)
        parts={
            'total':total.detach(), 'local_det':detector_loss.detach(),
            'align_weighted':align_loss.detach(),
            'roi_guide':guide_loss.detach(), 'roi_center':center_loss.detach(),
            'roi_positive':float(n_pos)/max(b,1),
            'roi_tir_valid':extracted['roi'].tir_valid.mean().detach(),
            'gt_assist':extracted['gt_assisted'].detach(),
        }
        parts.update(_extract_component_losses(items,'local'))
        for name,key in [('anchor','anchor'),('coarse','coarse'),('scale','scale'),
                         ('targetness','targetness'),('v2_quality','v2_quality'),
                         ('v2_route','v2_route'),('gmls_gt_quality','v2_gmls_gt_quality'),
                         ('dmls_gt_quality','v2_dmls_gt_quality')]:
            if key in align_stats:
                parts['align_'+name]=align_stats[key].detach()
        return total,parts

    @torch.no_grad()
    def predict(self,batch,*,conf=0.001,iou=0.70):
        if self.training:
            raise RuntimeError('Call model.eval() before predict')
        from engine.evaluator import postprocess_predictions
        ex=self.encode_and_crop(batch,return_debug=False,train_rois=False)
        raw=self.detector.predict(ex['rgb_roi'],ex['tir_roi'])
        loc=postprocess_predictions(raw,self.detector,conf_thres=conf,iou_thres=iou)
        result=[]
        for i,p in enumerate(loc):
            box=boxes_roi_to_semantic(p['bboxes'],ex['roi'],batch,i)
            result.append({'bboxes':box,'conf':p['conf'],'cls':p['cls']})
        return result,ex
