#!/usr/bin/env python3
# CPU regression test for independent P34-only variant (no Ultralytics needed).
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import os
import types

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
models = types.ModuleType('models')
models.__path__ = []
parent_mod = types.ModuleType('models.rgbt_rlsfa_gtga_centered_bhlr_model')

class FakeParent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fusion_type = 'add'
        self.alignment_enabled = True
        self.scale_to_index = {3:4, 4:6, 5:10}
        self.fake_backbone=nn.Conv2d(4,4,1)
    def forward_features(self, rgb, tir, **kwargs):
        b,c,h,w=rgb.shape
        rgb0=self.fake_backbone(rgb)
        out_rgb=[torch.zeros_like(rgb0) for _ in range(11)]
        out_fused=list(out_rgb)
        f={};g={}
        for s,idx in self.scale_to_index.items():
            out_rgb[idx]=rgb0
            out_fused[idx]=rgb0+tir
            f[idx]=rgb0+tir
            g[s]={
              'rgb_target_support':torch.ones(b,1,h,w),
              'aligned_tir_target_support':torch.full((b,1,h,w),0.7),
              'joint_target_support':torch.full((b,1,h,w),0.8),
            }
        return {'rgb_backbone':out_rgb,'fused_backbone':out_fused,'fusion_features':f,
                'alignment_info':{'guidance_by_scale':g,
                                  'coarse_confidence':torch.full((b,1,1,1),0.3)}}
parent_mod.RGBTRLSFAGTGACenteredBHLRV2DetectionModel=FakeParent
sys.modules.update({'models':models,'models.rgbt_rlsfa_gtga_centered_bhlr_model':parent_mod})

v1_path = ROOT / 'models/rgbt_reliability_fusion_model.py'
if not v1_path.exists():
    # Test in extracted patch directory: locate previously shipped v1 source.
    v1_path = ROOT.parent / 'arf_p34_source/models/rgbt_reliability_fusion_model.py'
    if not v1_path.exists():
        raise RuntimeError('ARF-v1 gate implementation missing. Install previous patch first.')

spec=importlib.util.spec_from_file_location('models.rgbt_reliability_fusion_model',v1_path)
v1=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=v1
spec.loader.exec_module(v1)

p34_path=ROOT/'models/rgbt_reliability_fusion_p34_model.py'
spec=importlib.util.spec_from_file_location('models.rgbt_reliability_fusion_p34_model',p34_path)
new=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=new
spec.loader.exec_module(new)

model=new.RGBTAlignmentReliabilityP34FusionDetectionModel(
    reliability_fusion_cfg={'enabled':True,'scales':[3,4],'hidden_channels':8})
assert list(model.reliability_gates)==['3','4']
assert all('reliability_gates.5' not in k for k in model.state_dict())
rgb=torch.randn(2,4,12,12,requires_grad=True)
tir=torch.randn(2,4,12,12,requires_grad=True)
out=model.forward_features(rgb,tir)
base=model.fake_backbone(rgb)
for s,idx in model.scale_to_index.items():
    assert torch.equal(out['fused_backbone'][idx],base+tir),(s,'not exact ADD at init')
loss=sum(out['fused_backbone'][i].square().mean() for i in (4,6,10))
loss.backward()
for s in (3,4):
    grads=model.reliability_gates[str(s)].net[-1].weight.grad
    assert grads is not None and grads.abs().sum().item()>0, (s,'no gradient')
with torch.no_grad():model.reliability_gates['3'].net[-1].bias.fill_(-2.)
out=model.forward_features(rgb.detach(),tir.detach())
assert not torch.allclose(out['fused_backbone'][4],model.fake_backbone(rgb.detach())+tir.detach())
assert torch.equal(out['fused_backbone'][10],model.fake_backbone(rgb.detach())+tir.detach())
model.eval()
os.environ['RGBT_RELIABILITY_EVAL_MODE']='force_add'
out=model.forward_features(rgb.detach(),tir.detach())
assert torch.equal(out['fused_backbone'][4],model.fake_backbone(rgb.detach())+tir.detach())
assert torch.equal(out['fused_backbone'][10],model.fake_backbone(rgb.detach())+tir.detach())
os.environ.pop('RGBT_RELIABILITY_EVAL_MODE')

p=ROOT/'scripts/install_arf_p34_v2.py'
spec=importlib.util.spec_from_file_location('patch_installer',p)
ins=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ins)
sample="""def build_model_from_config(variant, model_cfg):
    elif_placeholder = None
    elif_variant = None
    if variant in {
        "rlsfa_gtga_centered_bhlr_v2",
        "rlsfa_gtga_centered",
        "rlsfa_gtga_centered_reliability_fusion_v1",
    }:
        if variant == "rlsfa_gtga_centered_reliability_fusion_v1":
            from models.rgbt_reliability_fusion_model import RGBTAlignmentReliabilityFusionDetectionModel
            model_class = RGBTAlignmentReliabilityFusionDetectionModel
            model_extra = {"reliability_fusion_cfg": dict(model_cfg.get("reliability_fusion", {}) or {})}
        else:
            model_class = object
            model_extra = {}
        model = model_class(**model_extra)
    return model
"""
patched=ins.apply(sample)
compile(patched,'factory','exec')
assert patched==ins.apply(patched)
assert 'ARF_P34_FACTORY_V2' in patched
print('[PASS] P3/P4-only gates; P5 ADD; exact init; gradients; eval force_add; factory idempotence')
