import unittest
from types import SimpleNamespace
import torch
from torch import nn
from cdgr_gpcf.fusion import ProgressiveComplement
from cdgr_gpcf.geometry import roi_guidance
from cdgr_gpcf.progressive import run_progressive

torch.set_num_threads(1)

def layer_input(m,x,outputs):
    f=m.f
    return x if f==-1 else outputs[f] if isinstance(f,int) else [x if j==-1 else outputs[j] for j in f]
class Add(nn.Module):
    def forward(self,r,t):return r+t
class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.ModuleList(nn.Conv2d(4,4,1,bias=False) for _ in range(3));self.tir_backbone=nn.ModuleList(nn.Conv2d(4,4,1,bias=False) for _ in range(3));self.backbone_len=3
        for m in [*self.model,*self.tir_backbone]:m.f=-1;nn.init.dirac_(m.weight)
        self.fusion_indices=[0,1,2];self.gpcf_blocks=nn.ModuleDict({'0':ProgressiveComplement(4,4,True,False,1.)});self._gpcf_guides={};self._gpcf_keep=None;self.gpcf_intervention='normal'
        with torch.no_grad():self.gpcf_blocks['0'].out_r.weight.fill_(.3);self.gpcf_blocks['0'].out_t.weight.fill_(.3)
    def fuse_backbone_features(self,r,t):
        f=[a+b for a,b in zip(r,t)];return f,dict(enumerate(f))

class Tests(unittest.TestCase):
    def test_shapes_gradients(self):
        for h in (80,40):
            m=ProgressiveComplement(512,64,True,True);r=torch.randn(1,512,h,h,requires_grad=True);t=torch.randn_like(r,requires_grad=True);g=torch.ones(1,4,h,h,requires_grad=True)
            a,b=m(r,t,g);self.assertEqual(a.shape,r.shape);(a.square().mean()+b.square().mean()).backward()
            self.assertTrue(torch.isfinite(r.grad).all());self.assertIsNotNone(m.out_r.weight.grad);self.assertIsNotNone(m.out_t.weight.grad);self.assertIsNone(g.grad)
    def test_mask(self):
        m=ProgressiveComplement(8,4,True,True);r=torch.randn(2,8,8,8);t=torch.randn_like(r);g=torch.ones(2,4,8,8);g[:,3,:,:4]=0
        a,b=m(r,t,g);self.assertTrue(torch.equal(a[:,:,:,:4],r[:,:,:,:4]));self.assertTrue(torch.equal(b[:,:,:,:4],t[:,:,:,:4]));self.assertGreater((a-r).abs().sum(),0)
    def test_off_and_direction(self):
        m=ProgressiveComplement(8,4,True,False);r=torch.randn(1,8,8,8);t=torch.randn_like(r);m.intervention='off';a,b=m(r,t);self.assertTrue(torch.equal(a,r));self.assertTrue(torch.equal(b,t));m.intervention='no_t_to_r';a,b=m(r,t);self.assertTrue(torch.equal(a,r));self.assertGreater((b-t).abs().sum(),0)
    def test_joint(self):
        m=ProgressiveComplement(8,4,False,False);r=torch.randn(1,8,8,8);t=torch.randn_like(r);a,b=m(r,t);torch.testing.assert_close(a-r,b-t,atol=1e-6,rtol=1e-4)
    def test_progression(self):
        det=Toy();r=torch.randn(1,4,8,8);t=torch.randn_like(r);# vary output by channel, since normalization yields approximately zero overall mean
        with torch.no_grad():nn.init.dirac_(det.gpcf_blocks['0'].out_r.weight);nn.init.dirac_(det.gpcf_blocks['0'].out_t.weight)
        normal=run_progressive(det,r,t,layer_input);det.gpcf_blocks['0'].intervention='off';off=run_progressive(det,r,t,layer_input)
        torch.testing.assert_close(off['fused_backbone'][2],r+t);self.assertGreater((normal['rgb_backbone'][2]-off['rgb_backbone'][2]).abs().sum(),0)
        self.assertGreater((normal['tir_backbone'][2]-off['tir_backbone'][2]).abs().sum(),0)
    def test_geometry(self):
        b=1;k=2;batch={'rgb_img':torch.zeros(b,3,24,24),'rgb_semantic_img':torch.zeros(b,3,8,8),'rgb_ratio_pad':[{'ratio':3.,'pad':[0,0]}],'rgb_semantic_ratio_pad':[{'ratio':1.,'pad':[0,0]}]}
        yy,xx=torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,8),indexing='ij');support=torch.stack((xx,yy,(xx+yy)/2))[None].requires_grad_()
        ex={'alignment_info':{'multi_target_guidance':support,'sampling_scale':torch.ones(1,2,requires_grad=True),'sampling_translation':torch.zeros(1,2,requires_grad=True)},'rois_per_image':k,'roi':SimpleNamespace(xy0=torch.tensor([[0.,0.],[12.,12.]],requires_grad=True),size=8)}
        g=roi_guidance(batch,ex,(4,4));self.assertEqual(g.shape,(2,4,4,4));self.assertFalse(g.requires_grad);self.assertTrue((g[:,3]==1).all());self.assertGreater(g[1,0].mean(),g[0,0].mean());self.assertAlmostEqual(float(g[0,0,0,0]),.5/21,places=5)
        ex['alignment_info']['sampling_translation']=torch.ones(1,2)*2;self.assertEqual(float(roi_guidance(batch,ex,(4,4))[:,3].sum()),0.)
    def test_baseline_init_small(self):
        m=ProgressiveComplement(512,64,True,True);r=torch.randn(1,512,8,8);t=torch.randn_like(r);a,b=m(r,t,torch.ones(1,4,8,8));self.assertLess(float((a-r).detach().abs().mean()),.01);self.assertLess(float((b-t).detach().abs().mean()),.01)

if __name__=='__main__':unittest.main()
