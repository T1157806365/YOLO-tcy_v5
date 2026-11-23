"""CPU-only tests; run from project root with `python -m unittest discover ...`."""
import unittest
from types import SimpleNamespace
import torch
from cdgr_garf.fusion import CDGRAsymmetricFusion,make_cdgr_roi_context


class TestGARF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def make_context(self,b=2):
        return torch.randn(b,7)*0.1

    def test_exact_add_at_initialization_spatial_and_channel(self):
        for spatial in (True,False):
            m=CDGRAsymmetricFusion(32,7,8,spatial=spatial).eval()
            x=torch.randn(2,32,12,12)
            y=torch.randn_like(x)
            m.set_roi_context(self.make_context())
            z=m(x,y)
            self.assertTrue(torch.equal(z,x+y),f'Spatial={spatial} init not exactly ADD')

    def test_gradient_to_gate_and_modalities(self):
        m=CDGRAsymmetricFusion(24,7,8,spatial=True).train()
        x=torch.randn(2,24,10,10,requires_grad=True)
        y=torch.randn_like(x,requires_grad=True)
        geo=torch.randn(2,7,requires_grad=True)
        m.set_roi_context(geo)
        z=m(x,y)
        z.square().mean().backward()
        self.assertIsNotNone(m.channel_gate[-1].weight.grad)
        self.assertGreater(float(m.channel_gate[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(m.spatial_gate[-1].weight.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertTrue(torch.isfinite(geo.grad).all())

    def test_geometry_steers_gate_when_learned(self):
        m=CDGRAsymmetricFusion(16,7,8,spatial=False).eval()
        x=torch.randn(2,16,8,8)
        y=torch.randn_like(x)
        # At initialization geometry cannot change F=R+T.
        c1=torch.zeros(2,7)
        c2=torch.ones(2,7)
        m.set_roi_context(c1)
        a=m(x,y)
        m.set_roi_context(c2)
        b=m(x,y)
        self.assertTrue(torch.equal(a,b))
        # One tiny optimizer-like update enables geometry conditioning.
        with torch.no_grad():
            m.channel_gate[-1].weight[:, -7:].fill_(0.08)
        m.set_roi_context(c1)
        a=m(x,y)
        m.set_roi_context(c2)
        b=m(x,y)
        self.assertGreater(float((a-b).abs().max().detach()),0.01)

    def test_context_gradient_scale_zero(self):
        m=CDGRAsymmetricFusion(16,7,8,spatial=False,geometry_grad_scale=0.0)
        with torch.no_grad():
            m.channel_gate[-1].weight[:, -7:].fill_(0.08)
        geo=torch.randn(2,7,requires_grad=True)
        x=torch.randn(2,16,8,8,requires_grad=True)
        y=torch.randn_like(x)
        m.set_roi_context(geo)
        m(x,y).square().mean().backward()
        self.assertTrue(geo.grad is None or float(geo.grad.abs().sum())==0.0)

    def test_context_from_cdgr_refinement_top2(self):
        b=2
        raw_s=torch.ones(b,2,requires_grad=True)
        raw_t=torch.zeros(b,2,requires_grad=True)
        delta=torch.tensor([[0.01,-0.02],[0.03,0.01]],requires_grad=True)
        refined_s=raw_s*torch.tensor([[1.1,0.95],[0.9,1.03]])
        a=torch.tensor([[0.25,0.5],[0.7,0.6]])
        refined_t=raw_s*a+raw_t+delta-refined_s*a
        ex={'alignment_info':{'sampling_scale':refined_s,'sampling_translation':refined_t,
                              'sampling_scale_raw_gtga':raw_s,'sampling_translation_raw_gtga':raw_t,
                              'coarse_anchor':a},'rois_per_image':2,
            'roi':SimpleNamespace(tir_valid=torch.tensor([1.,.9,.8,.7]))}
        batch={'rgb_img':torch.zeros(b,3,192,192),
               'rgb_semantic_img':torch.zeros(b,3,64,64),
               'tir_img':torch.zeros(b,3,64,64)}
        c=make_cdgr_roi_context(batch,ex,max_shift_px=24,max_log_scale=.5)
        self.assertEqual(c.shape,(4,7))
        self.assertTrue(torch.allclose(c[0,:2],delta[0]*63/24,atol=1e-6))
        self.assertTrue(torch.allclose(c[0,:6],c[1,:6]))
        self.assertNotEqual(float(c[0,-1].detach()),float(c[1,-1].detach()))
        c.sum().backward()
        self.assertGreater(float(delta.grad.abs().sum()),0)

    def test_local_detection_can_backpropagate_to_geometry_after_learning(self):
        m=CDGRAsymmetricFusion(16,7,8,spatial=False,geometry_grad_scale=0.1).eval()
        # The last layer starts at zero (exact Add). After an update, geometry
        # should receive a controlled gradient from downstream detection loss.
        with torch.no_grad():
            m.channel_gate[-1].weight.normal_(0,0.01)
        geo=torch.randn(2,7,requires_grad=True)
        x=torch.randn(2,16,8,8,requires_grad=True)
        y=torch.randn_like(x)
        m.set_roi_context(geo)
        m(x,y).square().mean().backward()
        self.assertIsNotNone(geo.grad)
        self.assertGreater(float(geo.grad.abs().sum()),0.0)
        self.assertTrue(torch.isfinite(geo.grad).all())

    def test_numerical_stability_and_cleanup(self):
        m=CDGRAsymmetricFusion(16,7,8,spatial=True).train()
        x=torch.randn(2,16,16,16)
        y=torch.randn_like(x)
        m.set_roi_context(self.make_context())
        out=m(x,y)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(all(torch.isfinite(v).all() for v in m.last_stats.values()))
        m.set_roi_context(None)
        self.assertIsNone(m._roi_context)
        with self.assertRaisesRegex(RuntimeError,'context missing'):
            m(x,y)


if __name__=='__main__':unittest.main()
