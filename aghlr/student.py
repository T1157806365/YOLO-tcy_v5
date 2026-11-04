"""Integrate AGHLR in original YOLO-tcy_v5 without changing core source.

Only the final Detect box branch P3 raw outputs are changed; classification,
P4, P5, GMLS/DMLS geometry, original teacher init and train/val remain intact.
The original BHLR *feature injection* is bypassed, but its detail encoder,
selector and learned alignment guidance still run and receive regression grads.
"""
from __future__ import annotations

import os
import torch
from torch import nn

from ga_monoalign_pcgrad.model import AlignmentProtectedStudent
from .adapter import AGHLRP3RegAdapter


def _first_conv(module: nn.Module) -> nn.Conv2d:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    raise RuntimeError('YOLO26 P3 box head has no Conv2d; unsupported version')


class AGHLRStudent(AlignmentProtectedStudent):
    def __init__(self, core, cfg):
        super().__init__(core, cfg)
        opts = cfg.get('aghlr', {}) or {}
        if not bool(opts.get('enabled', False)):
            raise ValueError('AGHLRStudent requires aghlr.enabled: true')
        if not core.bhlr_enabled or core.bhlr is None or core.bhlr_scales != (3,):
            raise RuntimeError('AGHLR requires original BHLR-v2 P3 enabled')
        if core.bhlr_guidance_mode != 'alignment' or not core.alignment_enabled:
            raise RuntimeError('AGHLR requires original GMLS alignment guidance')
        self._aghlr_mode = os.environ.get('AGHLR_EVAL_MODE','on').strip().lower()
        if self._aghlr_mode not in ('on','off'):
            raise ValueError('AGHLR_EVAL_MODE must be on or off')
        detect = core.model[-1]
        if int(getattr(detect, 'reg_max', -1)) != 1:
            raise RuntimeError('AGHLR requires YOLO26 DFL-free reg_max=1; refusing to modify DFL bins')
        if not hasattr(detect,'cv2') or detect.cv2 is None or len(detect.cv2) < 3:
            raise RuntimeError('Unsupported YOLO Detect: missing cv2 three-scale box branch')
        if bool(getattr(detect,'end2end',False)) and not hasattr(detect,'one2one_cv2'):
            raise RuntimeError('Missing one2one_cv2 for YOLO26 end-to-end head')
        neck_channels = int(_first_conv(detect.cv2[0]).in_channels)
        detail_channels = int(cfg['model']['bhlr']['detail_channels'])
        r = int(opts.get('subcell_ratio',4))
        if int(cfg['data']['rgb_imgsz']) != 1920 or int(cfg['data']['rgb_semantic_imgsz']) != 640:
            raise RuntimeError('AGHLR release is specific to RGB high 1920 / semantic 640')
        if 320 % r:
            raise RuntimeError('subcell_ratio must divide BHLR 320 grid')
        sampling_mode = os.environ.get(
            'AGHLR_SAMPLING_MODE', str(opts.get('sampling_mode', 'legacy'))
        ).strip().lower()
        self.aghlr = AGHLRP3RegAdapter(
            neck_channels=neck_channels,
            detail_channels=detail_channels,
            hidden_channels=int(opts.get('hidden_channels',24)),
            pre_s2d_channels=int(opts.get('pre_s2d_channels',8)),
            subcell_ratio=r,
            max_offset=float(opts.get('max_offset',0.50)),
            sampling_mode=sampling_mode,
            sampling_radius_p3=int(opts.get('sampling_radius_p3',2)),
            sampling_temperature=float(opts.get('sampling_temperature',10.0)),
            sampling_distance_penalty=float(opts.get('sampling_distance_penalty',0.75)),
            sampling_max_shift_p3=float(opts.get('sampling_max_shift_p3',0.75)),
            sampling_gain=float(opts.get('sampling_gain',0.75)),
        )
        self._aghlr_data = {}  # ephemeral tensors, NEVER part of checkpoint
        self._aghlr_last = {}
        self._aghlr_handles = []
        self._install_hooks(detect)

    def _install_hooks(self, detect):
        core=self.core
        p3_backbone_id = int(core.scale_to_index[3])
        bhlr_adapter = core.bhlr.adapters[str(p3_backbone_id)]
        guide_fusion = core.alignment_guidance_fusion['3']

        def detail_hook(module, inputs, output):
            self._aghlr_data['detail'] = output

        def selector_hook(module, inputs, output):
            if not isinstance(output,(tuple,list)) or len(output) < 2:
                raise RuntimeError('BHLR P3 selector output signature changed')
            self._aghlr_data['selector'] = output[1]

        def guide_hook(module, inputs, output):
            if not isinstance(output,(tuple,list)) or len(output)<1:
                raise RuntimeError('Original alignment guidance output changed')
            self._aghlr_data['support'] = output[0]

        def remove_old_injection(module, inputs, output):
            # BHLRScaleAdapter.forward(detail_raw, rgb_feat, tir_feat, ...).
            # Keep the detail calculation, but route HR information ONLY to
            # Detect P3 LTRB regression logits, not into whole RGB P3 features.
            if len(inputs) < 2 or not isinstance(inputs[1],torch.Tensor):
                raise RuntimeError('BHLR adapter RGB feature argument has changed')
            raw_rgb_p3 = inputs[1]
            if isinstance(output,tuple):
                if len(output) != 2 or not isinstance(output[1],dict):
                    raise RuntimeError('Unexpected BHLR debug return')
                debug = dict(output[1]);debug['enhanced_rgb']=raw_rgb_p3
                return raw_rgb_p3, debug
            if not isinstance(output,torch.Tensor):
                raise RuntimeError('Unexpected BHLR P3 return type')
            return raw_rgb_p3

        def head_pre_hook(module, inputs):
            if self._aghlr_mode=='off':
                self._aghlr_data.pop('delta', None)
                return None
            if not inputs or not isinstance(inputs[0],(list,tuple)) or len(inputs[0])<3:
                raise RuntimeError('Detect expected a three-scale feature list')
            missing = [k for k in ('detail','selector','support') if k not in self._aghlr_data]
            if missing:
                raise RuntimeError('AGHLR missing BHLR features: '+str(missing))
            neck_p3 = inputs[0][0]
            delta = self.aghlr(neck_p3,
                         self._aghlr_data['detail'],
                         self._aghlr_data['support'],
                         self._aghlr_data['selector'])
            self._aghlr_data['delta'] = delta
            self._aghlr_last.update(self.aghlr.last_sampling_stats)
            with torch.no_grad():
                self._aghlr_last['delta_rms'] = float(delta.float().square().mean().sqrt().detach().cpu())
                self._aghlr_last['delta_max_abs'] = float(delta.float().abs().max().detach().cpu())
            return None

        def box_hook_one2many(module, inputs, output):
            d=self._aghlr_data.get('delta',None)
            if d is None: return output
            if not torch.is_tensor(output) or output.shape != d.shape:
                raise RuntimeError(f'AGHLR P3 cv2 mismatch: logits {getattr(output,"shape",None)}, delta {d.shape}')
            return output + d

        def box_hook_one2one(module, inputs, output):
            d=self._aghlr_data.get('delta',None)
            if d is None:return output
            if not torch.is_tensor(output) or output.shape != d.shape:
                raise RuntimeError(f'AGHLR P3 one2one mismatch: {getattr(output,"shape",None)} vs {d.shape}')
            # Match Ultralytics' one2one x.detach() intent: one2one losses
            # should not train the shared backbone/detail pathway directly.
            return output + (d.detach() if module.training else d)

        def head_cleanup(module, inputs, output):
            # Do not retain a stale autograd graph between training batches.
            self._aghlr_data.clear()

        self._aghlr_handles.extend([
            core.bhlr.detail_stem.register_forward_hook(detail_hook),
            bhlr_adapter.selector.register_forward_hook(selector_hook),
            guide_fusion.register_forward_hook(guide_hook),
            bhlr_adapter.register_forward_hook(remove_old_injection),
            detect.register_forward_pre_hook(head_pre_hook),
            detect.cv2[0].register_forward_hook(box_hook_one2many),
            detect.register_forward_hook(head_cleanup),
        ])
        if bool(getattr(detect,'end2end',False)):
            self._aghlr_handles.append(detect.one2one_cv2[0].register_forward_hook(box_hook_one2one))

    def loss(self,batch):
        if self._aghlr_mode=='off':
            raise RuntimeError('AGHLR_EVAL_MODE=off is for evaluation/ablation only, not training')
        out, logs = super().loss(batch)
        logs['aghlr_delta_rms'] = self._aghlr_last.get('delta_rms',0.0)
        logs['aghlr_delta_max_abs'] = self._aghlr_last.get('delta_max_abs',0.0)
        logs['aghlr_sampling_shift_p3'] = self._aghlr_last.get('mean_shift_p3',0.0)
        return out,logs
