import unittest
import torch
from torch import nn
from types import SimpleNamespace
from cdgr_smf.fusion import ScaleMatchedAdd,match_grid,fuse_scale_matched
from gmls_roi_hybrid.geometry import sample_roi_pair

class ScaleTest(unittest.TestCase):
    def setUp(self):torch.manual_seed(0)
    def test_equal_grid_identity(self):
        r=torch.randn(2,8,16,16);t=torch.randn_like(r)
        self.assertIs(match_grid(t,r),t)
        self.assertTrue(torch.equal(ScaleMatchedAdd(8,8)(r,t),r+t))
    def test_projection_gradient(self):
        r=torch.randn(2,8,16,16,requires_grad=True);t=torch.randn(2,4,16,16,requires_grad=True)
        f=ScaleMatchedAdd(8,4);z=f(r,t);z.square().mean().backward()
        self.assertEqual(z.shape,r.shape)
        for v in (r.grad,t.grad,f.tir_proj.weight.grad):self.assertTrue(torch.isfinite(v).all());self.assertGreater(v.abs().sum().item(),0)
    def test_origin_grid_resize(self):
        # 56 -> 80 origin-aligned resampling, compare a linear ramp.
        t=torch.arange(56,dtype=torch.float32)[None,None,None,:].expand(1,1,56,56)
        z=match_grid(t,torch.zeros(1,1,80,80))
        target=(torch.arange(80)*56/80).clamp(max=55)
        torch.testing.assert_close(z[0,0,20],target,atol=1e-5,rtol=1e-5)
    def test_pyramid_routing_and_gradients(self):
        for n in (320,224):
            r=[None]*11;t=[None]*7
            for i,c,sz in [(4,16,80),(6,16,40),(10,16,20)]:r[i]=torch.randn(1,c,sz,sz,requires_grad=True)
            for i,c,sz in [(2,8,n//4),(4,16,n//8),(6,16,n//16)]:t[i]=torch.randn(1,c,sz,sz,requires_grad=True)
            det=SimpleNamespace(smf_pairs=[(4,2),(6,4),(10,6)],fusions=nn.ModuleDict({'4':ScaleMatchedAdd(16,8),'6':ScaleMatchedAdd(16,16),'10':ScaleMatchedAdd(16,16)}))
            outs,features=fuse_scale_matched(det,r,t)
            self.assertEqual([outs[i].shape for i in (4,6,10)],[r[i].shape for i in (4,6,10)])
            sum(v.square().mean() for v in features.values()).backward()
            for i in (2,4,6):self.assertGreater(t[i].grad.abs().sum().item(),0)
            self.assertEqual(len(det.smf_stats),12)
    def test_direct_crop_320_and_original(self):
        # Direct 320 sampling must equal every second point in original 640 grid.
        h=768;s=256
        rgb=torch.zeros(1,3,h,h);tir=torch.rand(1,3,s,s)
        center=torch.tensor([[.5,.5]]);scale=torch.tensor([[.9,1.1]]);trans=torch.tensor([[.03,-.02]])
        hi=[{'ratio':3.,'pad':(0.,0.)}];sem=[{'ratio':1.,'pad':(0.,0.)}]
        args=(rgb,tir,center,scale,trans,hi,sem,(s,s))
        r0,t0,g0=sample_roi_pair(*args,size=640)
        r1,t1,g1=sample_roi_pair(*args,size=640,tir_size=320)
        torch.testing.assert_close(r1,r0,rtol=0,atol=0);torch.testing.assert_close(g1.xy0,g0.xy0)
        torch.testing.assert_close(t1,t0[:,:,::2,::2],rtol=1e-5,atol=1e-5)
    def test_crop_geometry_gradient(self):
        rgb=torch.rand(1,3,768,768);tir=torch.rand(1,3,256,256)
        center=torch.tensor([[.5,.5]],requires_grad=True);scale=torch.tensor([[.9,1.1]],requires_grad=True);trans=torch.tensor([[.03,-.02]],requires_grad=True)
        r,t,g=sample_roi_pair(rgb,tir,center,scale,trans,[{'ratio':3.,'pad':(0.,0.)}],[{'ratio':1.,'pad':(0.,0.)}],(256,256),tir_size=224)
        self.assertEqual(t.shape[-2:],(224,224));self.assertEqual(r.shape[-2:],(640,640))
        (r.square().mean()+t.square().mean()).backward()
        for v in (center.grad,scale.grad,trans.grad):self.assertTrue(torch.isfinite(v).all());self.assertGreater(v.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
