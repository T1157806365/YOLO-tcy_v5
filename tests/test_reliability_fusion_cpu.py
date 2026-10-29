#!/usr/bin/env python3
"""CPU interface/gradient test without user's model files or GPU."""
from __future__ import annotations
import sys
import types
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

module_base=types.ModuleType('models')
module_base.__path__=[]
module_parent=types.ModuleType('models.rgbt_rlsfa_gtga_centered_bhlr_model')
class FakeParent(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.fusion_type='add'
        self.alignment_enabled=True
        self.scale_to_index={3:4,4:6,5:10}
        self.fake_backbone=nn.Conv2d(4,4,1)
    def forward_features(self,rgb,tir,**kwargs):
        b,c,h,w=rgb.shape
        rgb0=self.fake_backbone(rgb)
        t=tir
        out_rgb=[torch.zeros_like(rgb0) for _ in range(11)]
        out_fused=list(out_rgb)
        out_t=[torch.zeros_like(t) for _ in range(11)]
        f={}
        guide={}
        for s,idx in self.scale_to_index.items():
            out_rgb[idx]=rgb0
            out_t[idx]=t
            out_fused[idx]=rgb0+t
            f[idx]=rgb0+t
            guide[s]={
              'rgb_target_support':torch.ones(b,1,h,w),
              'aligned_tir_target_support':torch.full((b,1,h,w),0.7),
              'joint_target_support':torch.full((b,1,h,w),0.8),
            }
        return {'rgb_backbone':out_rgb,'aligned_tir_backbone':out_t,
                'fused_backbone':out_fused,'fusion_features':f,
                'alignment_info':{'guidance_by_scale':guide,'coarse_confidence':torch.full((b,1,1,1),0.3)}}
module_parent.RGBTRLSFAGTGACenteredBHLRV2DetectionModel=FakeParent
sys.modules.update({'models':module_base,'models.rgbt_rlsfa_gtga_centered_bhlr_model':module_parent})

file=Path(__file__).resolve().parents[1]/'models/rgbt_reliability_fusion_model.py'
spec=importlib.util.spec_from_file_location('test_custom_model',file)
new=importlib.util.module_from_spec(spec)
spec.loader.exec_module(new)
model=new.RGBTAlignmentReliabilityFusionDetectionModel(reliability_fusion_cfg={
    'enabled':True,'hidden_channels':8,'detach_alignment_support':True})

rgb=torch.randn(2,4,12,12,requires_grad=True)
tir=torch.randn(2,4,12,12,requires_grad=True)
out=model.forward_features(rgb,tir)
base=model.fake_backbone(rgb)
for s,idx in model.scale_to_index.items():
    delta=(out['fused_backbone'][idx]-(base+tir)).abs().max().item()
    assert delta==0.0,(s,delta)
    assert torch.equal(out['reliability_modulation_maps'][s],torch.ones_like(out['reliability_modulation_maps'][s]))

# Gradients to the NEW fusion head MUST exist after a backward pass.
loss=sum(out['fused_backbone'][idx].square().mean() for idx in (4,6,10))
loss.backward()
for scale,gate in model.reliability_gates.items():
    assert gate.net[-1].weight.grad is not None,scale
    assert gate.net[-1].weight.grad.abs().sum().item()>0,scale

# Gate must have nontrivial effect after parameters deviate from identity.
with torch.no_grad():
    model.reliability_gates['3'].net[-1].bias.fill_(-2.0)
changed=model.forward_features(rgb.detach(),tir.detach())
assert not torch.allclose(changed['fused_backbone'][4],base.detach()+tir.detach())
assert torch.allclose(changed['fused_backbone'][6],base.detach()+tir.detach())

# eval override must EXACTLY revert to original ADD
import os
model.eval()
os.environ['RGBT_RELIABILITY_EVAL_MODE']='force_add'
forced=model.forward_features(rgb.detach(),tir.detach())
assert torch.equal(forced['fused_backbone'][4],(base+tir).detach())
del os.environ['RGBT_RELIABILITY_EVAL_MODE']
print('[PASS] exact ADD initialization; trainable gradient to each P3/P4/P5 gate; '
      'changed output after learning; force_add switch')

# Factory installation parse and idempotence, without mutating repository files.
p=Path(__file__).resolve().parents[1]/'scripts/install_reliability_fusion.py'
spec=importlib.util.spec_from_file_location('test_installer',p)
inst=importlib.util.module_from_spec(spec)
spec.loader.exec_module(inst)
sample='''def build_model_from_config(variant, model_cfg):
    if variant in {"x"}:
        model = 0
    elif variant in {
        "rlsfa_gtga_centered_bhlr_v2",
        "rlsfa_gtga_centered",
    }:
        from models.foo import RGBTRLSFAGTGACenteredBHLRV2DetectionModel
        model = RGBTRLSFAGTGACenteredBHLRV2DetectionModel(
            model_name="yolo26m",
            bhlr_enabled=True,
            verbose=verbose,
        )
    elif variant in {"other"}:
        model = 0
    return model
'''
newtext=inst.apply(sample)
assert newtext==inst.apply(newtext)
compile(newtext,'patched_factory','exec')
assert 'RL_FUSION_INTEGRATED_V1' in newtext
print('[PASS] installer syntax / idempotence')
