"""CPU tests for local feature correspondence fusion; no YOLO/GPU needed."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from cdgr_lcmf.fusion import LocalCorrespondenceFusion,make_cdgr_roi_context

class TestLCMF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def _layer(self,radius=2,channels=16):
        m=LocalCorrespondenceFusion(channels=channels,proj=8,hidden=16,
                                    radius=radius,stride=8)
        m.set_roi_context(torch.randn(2,7))
        return m

    def test_exact_old_add_all_scales(self):
        for radius in (0,1,2):
            m=self._layer(radius=radius).eval()
            r=torch.randn(2,16,12,12)
            t=torch.randn_like(r)
            with torch.no_grad(): out=m(r,t)
            self.assertTrue(torch.equal(out,r+t), f'radius {radius} not identical')

    def test_finite_outputs_and_monitor(self):
        m=self._layer().train()
        r=torch.randn(2,16,10,11,requires_grad=True)
        t=torch.randn_like(r,requires_grad=True)
        out=m(r,t)
        self.assertEqual(out.shape,r.shape)
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertEqual(set(m.last_stats),{
            'flow_px','peak_prob','resample_mix','tir_resample_delta',
            'interaction_ratio','effective_delta_ratio'})
        for v in m.last_stats.values():
            self.assertTrue(torch.isfinite(v).all().item())
        self.assertIsNone(m._context.grad)

    def test_grad_reaches_new_fusion_on_step_one(self):
        m=self._layer().train()
        rgb=torch.randn(2,16,8,8,requires_grad=True)
        tir=torch.randn_like(rgb,requires_grad=True)
        out=m(rgb,tir)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(rgb.grad).all().item())
        self.assertTrue(torch.isfinite(tir.grad).all().item())
        self.assertTrue(m.interaction[-1].weight.grad.abs().sum().item()>0)
        self.assertIsNotNone(m.raw_resample_mix.grad)
        self.assertTrue(torch.isfinite(m.raw_resample_mix.grad).item())

    def test_local_correspondence_learns_after_nonzero_mix(self):
        m=self._layer().train()
        with torch.no_grad():
            m.raw_resample_mix.fill_(0.1)
        rgb=torch.randn(2,16,7,7,requires_grad=True)
        tir=torch.randn_like(rgb,requires_grad=True)
        m(rgb,tir).square().mean().backward()
        self.assertTrue(torch.isfinite(m.rgb_embed[0].weight.grad).all().item())
        self.assertTrue(m.rgb_embed[0].weight.grad.abs().sum().item()>0)
        self.assertTrue(m.geo_bias[-1].weight.grad.abs().sum().item()>0)

    def test_correspondence_grid_direction(self):
        # Positive feature-x flow samples from x+1 in TIR.
        t=torch.arange(5,dtype=torch.float32).view(1,1,1,5).expand(1,1,3,5)
        flow=torch.zeros(1,3,5,2);flow[...,0]=1
        got=LocalCorrespondenceFusion._sample_offset(t,flow)
        self.assertTrue(torch.allclose(got[0,0,0],torch.tensor([1.,2.,3.,4.,4.])))

    def test_geometry_context_no_gt(self):
        b=2;k=2
        batch={'rgb_img':torch.zeros(b,3,192,192),'rgb_semantic_img':torch.zeros(b,3,64,64),
               'tir_img':torch.zeros(b,3,64,64)}
        class Ro:pass
        ro=Ro();ro.tir_valid=torch.tensor([1,.8,.9,.7])
        info={'sampling_scale':torch.ones(b,2),'sampling_translation':torch.zeros(b,2),
              'sampling_scale_raw_gtga':torch.ones(b,2),'sampling_translation_raw_gtga':torch.zeros(b,2),
              'coarse_anchor':torch.ones(b,2)*.5}
        ctx=make_cdgr_roi_context(batch,{'alignment_info':info,'roi':ro,'rois_per_image':k})
        self.assertEqual(tuple(ctx.shape),(4,7))
        self.assertTrue(torch.allclose(ctx[:,0:4],torch.zeros(4,4)))
        self.assertTrue(torch.allclose(ctx[:,-1],ro.tir_valid))

    def test_context_missing_error(self):
        m=LocalCorrespondenceFusion(channels=16,proj=8,hidden=16)
        with self.assertRaises(RuntimeError):m(torch.randn(2,16,5,5),torch.randn(2,16,5,5))

    def test_context_not_registered_in_state_dict(self):
        m=self._layer()
        state=m.state_dict()
        self.assertTrue(all('_context' not in k for k in state))

if __name__=='__main__':unittest.main()
