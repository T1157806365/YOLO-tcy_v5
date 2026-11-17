"""CPU mock integration and autocast tests without ultralytics dependencies."""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]

# Provide the exact imported name without loading unavailable ultralytics modules.
models = types.ModuleType("models"); models.__path__ = []
modules = types.ModuleType("models.modules"); modules.__path__ = []
stub = types.ModuleType("models.modules.bhlr_v2")


class StubEncoder(nn.Module):
    def __init__(self, ratio, detail_channels, stem_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3*ratio*ratio, stem_channels, 1), nn.ReLU(),
            nn.Conv2d(stem_channels, detail_channels, 3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)


stub.SharedLostDetailEncoder = StubEncoder
sys.modules.update({"models":models, "models.modules":modules,"models.modules.bhlr_v2":stub})
module_name = "models.modules.bhlr_dual_hr"
spec = importlib.util.spec_from_file_location(module_name, ROOT/"models/modules/bhlr_dual_hr.py")
mod = importlib.util.module_from_spec(spec);sys.modules[module_name]=mod;spec.loader.exec_module(mod)


class FakeBHLR(nn.Module):
    def __init__(self):
        super().__init__();self.ratio=3;self.scales=(4,)
        self.detail_stem=StubEncoder(3,48,16)

    def build_lost_detail_bank(self, rgb_high, rgb_semantic):
        rec = F.interpolate(rgb_semantic, size=rgb_high.shape[-2:], mode="bilinear", align_corners=False)
        residual = rgb_high-rec
        raw = self.detail_stem(F.pixel_unshuffle(residual,3))
        return raw,rec,residual

    def enhance_layer(self, layer_id, detail_raw, rgb_feat, tir_feat, external_guidance=None, return_debug=False):
        assert layer_id==4
        enh = rgb_feat + F.interpolate(detail_raw.mean(1, keepdim=True),size=rgb_feat.shape[-2:], mode="bilinear",align_corners=False)
        return (enh,{"original":True}) if return_debug else enh


class FakeHvad(nn.Module):
    def __init__(self):super().__init__();self._detail_bank=None


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__();self.bhlr=FakeBHLR();self.hvad=FakeHvad()
        self.bhlr_enabled=True;self.bhlr_scales=(3,);self.bhlr_guidance_mode="alignment";self.alignment=nn.Identity()

    def loss(self, batch, preds=None):
        return batch["loss"], {"box_loss": torch.tensor(1.)}


def guide(batch, size, requires_grad=False):
    d = {k:torch.rand(batch,1,size,size,requires_grad=requires_grad) for k in ("rgb_target_support","aligned_tir_target_support","original_joint_target_support")}
    return d


def run(mode):
    torch.manual_seed(2026)
    m=FakeModel()
    high=torch.rand(2,3,288,288)
    sem=torch.rand(2,3,96,96)
    native=m.bhlr.build_lost_detail_bank(high,sem)
    cfg={"enabled":True,"mode":mode,"detail_channels":48,"stem_channels":16,"ratio":3}
    m=mod.attach_dual_hr(m,cfg,verbose=False)
    raw,rec,res=m.bhlr.build_lost_detail_bank(high,sem)
    assert raw.shape==(2,48,48,48)
    assert res.shape==(2,3,288,288)
    assert not torch.allclose(native[2],res),"Clean residual should differ from independently resized reference"
    gt=guide(2,12,requires_grad=True)
    rgb=torch.rand(2,48,12,12)
    tir=torch.rand(2,48,12,12)
    # Identity initialized: no change to BHLR input at startup.
    ret=m.bhlr.enhance_layer(4,raw,rgb,tir,external_guidance=gt,return_debug=True)
    assert isinstance(ret,tuple) and len(ret)==2
    assert torch.equal(m.hvad._detail_bank,raw),"At init final bank should exactly equal residual bank"
    loss,items=m.loss({"loss":ret[0].square().mean()})
    assert "dualhr_gamma" in items
    assert float(items["dualhr_gamma"])==0
    loss.backward()
    if mode=="clean":
        assert m.bhlr.dual_hr.gamma_raw is None
    else:
        assert m.bhlr.dual_hr.gamma_raw.grad is not None
        with torch.no_grad():m.bhlr.dual_hr.gamma_raw.fill_(0.3)
        for p in m.parameters():
            if p.grad is not None: p.grad.zero_()
        raw,_,_=m.bhlr.build_lost_detail_bank(high,sem)
        fused=m.bhlr.enhance_layer(4,raw,rgb,tir,external_guidance=gt,return_debug=False)
        assert not torch.equal(m.hvad._detail_bank,raw),"Active structure should alter bank"
        fused.square().mean().backward()
        assert m.bhlr.dual_hr.structure_encoder.net[0].weight.grad is not None
        assert m.bhlr.dual_hr.selector[-1].weight.grad is not None
        assert torch.isfinite(m.bhlr.dual_hr.selector[-1].weight.grad).all()
    print("[PASS]",mode,"build / identity / fusion / HVAD bank / backward")


for mode in ("clean","dual","guided"):
    run(mode)

# CPU autocast with bfloat16: ensure spatial guidance math and gradients are stable.
m=mod.attach_dual_hr(FakeModel(),{"enabled":True,"mode":"guided","detail_channels":48,"stem_channels":16},verbose=False)
with torch.autocast(device_type="cpu",dtype=torch.bfloat16):
    hr=torch.rand(1,3,288,288)
    low=torch.rand(1,3,96,96)
    raw,_,_=m.bhlr.build_lost_detail_bank(hr,low)
    with torch.no_grad():m.bhlr.dual_hr.gamma_raw.fill_(0.2)
    y=m.bhlr.enhance_layer(4,raw,torch.rand(1,48,12,12),torch.rand(1,48,12,12),external_guidance=guide(1,12))
    z=y.square().mean()
z.backward()
assert torch.isfinite(z)
print("[PASS] CPU bfloat16 autocast forward/backward")
