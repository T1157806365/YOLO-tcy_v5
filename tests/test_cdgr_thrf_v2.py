import unittest
from types import SimpleNamespace
import torch
from cdgr_thrf_v2.fusion import PhasePreservingRetrieval,P3JointFusion,P2Context,THRFV2P3
from cdgr_thrf_v2.loss import local_thermal_targets,heat_loss

torch.set_num_threads(min(torch.get_num_threads(),2))
class TestTHRFV2(unittest.TestCase):
    def test_phase_order(self):
        x=torch.arange(16.).reshape(1,1,4,4)
        y=PhasePreservingRetrieval.phases(x)
        self.assertEqual(y[0,:,0,0,0].tolist(),[0.,1.,4.,5.])
        # Invert the pure reorganization, prior to learned projection.
        z=y.permute(0,2,1,3,4).reshape(1,4,2,2)
        self.assertTrue(torch.equal(x,torch.nn.functional.pixel_shuffle(z,2)))
    def test_qkv_gradient_and_condition(self):
        torch.manual_seed(3)
        m=PhasePreservingRetrieval(8,16,4,4,radius=1)
        r=torch.randn(2,8,8,8,requires_grad=True);t=torch.randn_like(r,requires_grad=True)
        t3=torch.randn(2,16,4,4,requires_grad=True);g=torch.zeros(2,7)
        d,h=m(r,t,t3,g);self.assertEqual(d.shape,(2,32,4,4));self.assertEqual(h.shape,(2,1,4,4))
        d2,_=m(r,torch.zeros_like(t),torch.zeros_like(t3),g)
        self.assertGreater(float((d-d2).detach().abs().mean()),1e-5)
        # Central phase entries unaffected by TIR query, retrieved ones change.
        dc=d.reshape(2,4,8,4,4);dc2=d2.reshape_as(dc)
        self.assertTrue(torch.equal(dc[:,:,:4],dc2[:,:,:4]))
        (d.square().mean()+h.square().mean()).backward()
        for x in (r,t,t3,m.q2.weight,m.q3.weight,m.k.weight,m.v.weight):
            self.assertIsNotNone(x.grad);self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
    def test_fusion_near_add_nonzero_detail_gradient(self):
        torch.manual_seed(4)
        m=P3JointFusion(16,32,groups=4,detail_init=.05)
        r=torch.randn(2,16,5,5,requires_grad=True);t=torch.randn_like(r,requires_grad=True);d=torch.randn(2,32,5,5,requires_grad=True)
        base=m(r,t,torch.zeros_like(d));self.assertTrue(torch.allclose(base,r+t,atol=1e-6))
        y=m(r,t,d);self.assertGreater(float((y-base).detach().abs().mean()),0.);self.assertLess(float((y-base).detach().abs().mean()),.3)
        y.square().mean().backward()
        for x in (r,t,d):self.assertGreater(x.grad.abs().sum().item(),0)
    def test_radius_zero(self):
        m=PhasePreservingRetrieval(8,16,4,4,radius=0)
        d,h=m(torch.randn(1,8,4,4),torch.randn(1,8,4,4),torch.randn(1,16,2,2),torch.zeros(1,7))
        self.assertTrue(torch.isfinite(d).all());self.assertAlmostEqual(float(m.last_stats['center_attention']),1.)
    def test_real_spatial_shape_backward(self):
        torch.manual_seed(2);ctx=P2Context();ctx.rgb=torch.randn(1,256,160,160,requires_grad=True);ctx.tir=torch.randn(1,256,160,160,requires_grad=True);ctx.geo=torch.zeros(1,7)
        f=THRFV2P3(ctx);r=torch.randn(1,512,80,80,requires_grad=True);t=torch.randn_like(r,requires_grad=True)
        out=f(r,t);self.assertEqual(out.shape,r.shape);self.assertTrue(torch.isfinite(out).all());(out.square().mean()+ctx.heat.square().mean()).backward()
        for x in (ctx.rgb,ctx.tir,r,t):self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(x.grad.abs().sum().item(),0)
        ctx.clear();self.assertIsNone(ctx.heat)
    def test_actual_tir_gt_inverse_mapping(self):
        # RGB high 1920, sem 640; TIR q=2*p+(-.3,-.3).  Construct actual TIR label
        # so inverse maps to the high-res crop center (960,960).
        b=3;k=2;rh=3.;rs=1.;meta=lambda r:{'ratio':r,'pad':[0.,0.]}
        high=torch.empty(b,3,1920,1920,device='meta');sem=torch.empty(b,3,640,640,device='meta');tir=torch.empty(b,3,640,640,device='meta')
        q=2*(960./3/639)-.3;label=q*639/640
        batch={'rgb_img':high,'rgb_semantic_img':sem,'tir_img':tir,'rgb_ratio_pad':[meta(rh)]*b,'rgb_semantic_ratio_pad':[meta(rs)]*b,
               'tir_bboxes':torch.tensor([[label,label,.05,.05]]),'tir_batch_idx':torch.tensor([0]),'rgb_batch_idx':torch.tensor([1])}
        # image0 TIR-only positive; image1 RGB-only ignore; image2 empty negative.
        roi=SimpleNamespace(xy0=torch.tensor([[640.,640.],[0.,0.]]*b),tir_valid=torch.ones(b*k),size=640)
        ex={'roi':roi,'rois_per_image':k,'alignment_info':{'sampling_scale':torch.full((b,2),2.),'sampling_translation':torch.full((b,2),-.3)}}
        boxes,idx,valid=local_thermal_targets(batch,ex)
        self.assertEqual(idx.tolist(),[0]);self.assertTrue(torch.allclose(boxes[0,:2],torch.tensor([.5,.5]),atol=1e-5));self.assertEqual(valid.tolist(),[True,True,False,False,True,True])
        h=torch.randn(b*k,1,8,8,requires_grad=True);loss,stats=heat_loss(h,boxes,idx,valid);loss.backward();self.assertTrue(torch.isfinite(h.grad).all());self.assertEqual(h.grad[2:4].abs().sum().item(),0.)
    def test_empty_ignored_heat_loss(self):
        h=torch.randn(2,1,4,4,requires_grad=True);loss,_=heat_loss(h,torch.empty(0,4),torch.empty(0,dtype=torch.long),torch.zeros(2,dtype=torch.bool));loss.backward();self.assertEqual(loss.item(),0.);self.assertTrue(torch.isfinite(h.grad).all())
if __name__=='__main__':unittest.main()
