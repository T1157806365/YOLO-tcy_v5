"""CPU tests for actual center/scale geometry, not a mocked tensor-gradient claim.

Only the imported legacy warp symbol is stubbed: the GTGA warp convention and
F.grid_sample are implemented with the same q = S*p+t coordinate system.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import torch
from torch import nn
from torch.nn import functional as F


def warp_with_geometry(x, scale, trans, output_hw=None, padding_mode='border', align_corners=True):
    b = x.shape[0]
    oh, ow = output_hw or x.shape[-2:]
    ys = torch.linspace(0, 1, oh, device=x.device)
    xs = torch.linspace(0, 1, ow, device=x.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    p = torch.stack((xx, yy), dim=-1)[None].expand(b,-1,-1,-1)
    q = p*scale.reshape(b,1,1,2)+trans.reshape(b,1,1,2)
    return F.grid_sample(x, q*2-1, align_corners=align_corners, padding_mode=padding_mode)

# Load standalone refiner without requiring ultralytics/Dataset on the CPU test host.
mod_name = 'models.modules.alignment.rlsfa_gtga_centered'
for name in ('models', 'models.modules', 'models.modules.alignment'):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules[mod_name] = types.ModuleType(mod_name)
sys.modules[mod_name].warp_with_geometry = warp_with_geometry
src = Path(__file__).resolve().parents[1]/'cdgr_geobox'/'refiner.py'
spec=importlib.util.spec_from_file_location('cdgr_refiner_under_test', src)
refiner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(refiner)


class DummyAligner(nn.Module):
    reference_scale = 3
    apply_scales = (3, 4, 5)
    padding_mode = 'border'
    align_corners = True
    def __init__(self):
        super().__init__()
        self.unit=types.SimpleNamespace(scale_min=0.5, scale_max=3.0)

    def align_pyramid(self, rgb_features, tir_features, scale_to_index,
                      rgb_valid_mask=None, tir_valid_mask=None, return_debug=False):
        b=rgb_features[0].shape[0]
        device=rgb_features[0].device
        scale=torch.tensor([[1.1,.9]],device=device).expand(b,2).clone()
        trans=torch.tensor([[.02,-.01]],device=device).expand(b,2).clone()
        anchor=torch.tensor([[.5,.45]],device=device).expand(b,2).clone()
        aligned=[]
        for scale_name in (3,4,5):
            i=scale_to_index[scale_name]
            aligned.append(warp_with_geometry(tir_features[i],scale,trans,rgb_features[i].shape[-2:]))
        guide=torch.rand(b,3,*rgb_features[0].shape[-2:],device=device)
        info=dict(sampling_scale=scale,sampling_translation=trans,coarse_anchor=anchor,
                  multi_target_guidance=guide,
                  support_by_scale={},guidance_by_scale={})
        return aligned,info


class TestCDGR(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.rgb=[torch.randn(2,8,n,n) for n in (24,12,6)]
        self.tir=[torch.randn(2,8,n,n) for n in (24,12,6)]
        self.conf={'p3_channels':8,'projection_channels':8,'hidden_channels':16,
                   'patch_size':7,'patch_radius':.12,'semantic_imgsz':192,
                   'max_shift_px':24,'max_log_scale':.50}
        self.module=refiner.CenterDecoupledAligner(DummyAligner(),self.conf)
        self.s2i={3:0,4:1,5:2}

    def test_zero_initial_residual_and_full_pyramid(self):
        aligned,info=self.module.align_pyramid(self.rgb,self.tir,self.s2i,return_debug=True)
        expected_s=torch.tensor([[1.1,.9],[1.1,.9]])
        expected_t=torch.tensor([[.02,-.01],[.02,-.01]])
        self.assertTrue(torch.allclose(info['sampling_scale'],expected_s,atol=1e-6))
        self.assertTrue(torch.allclose(info['sampling_translation'],expected_t,atol=1e-6))
        for level in range(3):
            manual=warp_with_geometry(self.tir[level],expected_s,expected_t,self.rgb[level].shape[-2:])
            self.assertTrue(torch.allclose(aligned[level],manual,atol=1e-5))
        self.assertEqual(info['multi_target_guidance'].shape,(2,3,24,24))

    def test_correction_preserves_anchor_pivot(self):
        with torch.no_grad():
            # Force a nonzero center and scale correction, independent of feature inputs.
            self.module.residual.regressor[-1].bias.copy_(torch.tensor([.3,-.2,.5,-.4]))
        _,info=self.module.align_pyramid(self.rgb,self.tir,self.s2i)
        anchor=info['coarse_anchor']
        old_q=info['sampling_scale_raw_gtga']*anchor+info['sampling_translation_raw_gtga']
        new_q=info['sampling_scale']*anchor+info['sampling_translation']
        expected_delta=torch.tanh(torch.tensor([.3,-.2]))*24.0/191.0
        self.assertTrue(torch.allclose(new_q-old_q,expected_delta.expand_as(new_q),atol=1e-5))
        self.assertTrue(torch.isfinite(info['sampling_scale']).all())

    def test_final_geo_gradients_reach_residual_head(self):
        _,info=self.module.align_pyramid(self.rgb,self.tir,self.s2i)
        # Use separate center and scale targets (as in GeoBox + FinalCenter).
        gt_s=torch.tensor([[1.35,.75],[1.3,.8]])
        gt_t=torch.tensor([[.045,-.025],[.04,-.03]])
        l=(info['sampling_scale']-gt_s).square().mean()+(info['sampling_translation']-gt_t).square().mean()
        l.backward()
        p=self.module.residual.regressor[-1]
        self.assertIsNotNone(p.weight.grad)
        self.assertGreater(p.weight.grad.abs().sum().item(),0)
        self.assertIsNotNone(self.module.residual.rgb_project[0].weight.grad)
        self.assertTrue(torch.isfinite(p.weight.grad).all())

    def test_no_gt_or_teacher_dependency(self):
        out,info=self.module(self.rgb,self.tir,self.s2i)
        self.assertEqual(len(out),3)
        self.assertNotIn('gt_box',info)
        self.assertFalse(any('teacher' in x for x in self.module.state_dict()))


if __name__=='__main__': unittest.main()
