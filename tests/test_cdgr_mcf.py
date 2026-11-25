import unittest,types
import torch
from cdgr_mcf.fusion import BidirectionalComplement,local_attention
from cdgr_mcf.targets import local_targets,subset_targets
from cdgr_mcf.monitor import hit_mask
from cdgr_mcf.heads import MonoHead
from torch import nn

torch.set_num_threads(2)
class TestMCF(unittest.TestCase):
    def test_attention_border(self):
        x=torch.zeros(1,4,3,3);v=torch.ones_like(x);out,a=local_attention(x,x,v)
        self.assertTrue(torch.allclose(out,v));self.assertTrue(torch.allclose(a.sum(1),torch.ones(1,3,3)))
        self.assertEqual(float(a[0,0,0,0]),0.)
    def test_same_scale_qkv_gradients(self):
        for hw in (80,40,20):
            m=BidirectionalComplement(512,16);r=torch.randn(1,512,hw,hw,requires_grad=True);t=torch.randn_like(r,requires_grad=True)
            y=m(r,t);self.assertEqual(y.shape,r.shape);(y.square().mean()).backward()
            for p in [r,t,*m.parameters()]:self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            for name in ('qr','qt','kr','kt','vr','vt'):
                self.assertGreater(float(getattr(m,name).weight.grad.abs().sum()),0.)
    def test_add_intervention(self):
        m=BidirectionalComplement(8,4);r=torch.randn(1,8,5,5);t=torch.randn_like(r);m.intervention='add'
        self.assertTrue(torch.equal(m(r,t),r+t))
    def test_cross_modal_sensitivity(self):
        m=BidirectionalComplement(8,4);r=torch.randn(1,8,5,5);t=torch.randn_like(r)
        y=m(r,t)-(r+t);z=m(r,t.flip(-1))-(r+t.flip(-1));self.assertGreater(float((y-z).detach().abs().mean()),1e-6)
    def batch(self):
        b=3;sz=640
        batch={'rgb_img':torch.empty(b,3,sz,sz),'rgb_semantic_img':torch.empty(b,3,sz,sz),'tir_img':torch.empty(b,3,sz,sz),
        'rgb_bboxes':torch.tensor([[.5,.5,.1,.1],[.5,.5,.1,.1]]),'rgb_batch_idx':torch.tensor([0,1]),'rgb_cls':torch.zeros(2,1),
        'tir_bboxes':torch.tensor([[.6,.5,.1,.1],[.6,.5,.1,.1]]),'tir_batch_idx':torch.tensor([0,2]),'tir_cls':torch.zeros(2,1),
        'rgb_ratio_pad':[{'ratio':1.,'pad':(0,0)}]*b,'rgb_semantic_ratio_pad':[{'ratio':1.,'pad':(0,0)}]*b}
        # q=p+.1 in x, so native TIR GT .6 maps to RGB .5, to ~1 pixel accuracy.
        ex={'roi':types.SimpleNamespace(xy0=torch.zeros(6,2),size=640,tir_valid=torch.ones(6)), 'rois_per_image':2,
            'alignment_info':{'sampling_scale':torch.ones(b,2),'sampling_translation':torch.tensor([[.1,0.]]*b)}}
        return batch,ex
    def test_real_tir_mapping_and_missing_labels(self):
        b,e=self.batch();t=local_targets(b,e,'tir');r=local_targets(b,e,'rgb')
        self.assertEqual(t['valid'].tolist(),[True,True,False,False,True,True]);self.assertEqual(r['valid'].tolist(),[True,True,True,True,False,False])
        self.assertEqual(t['idx'].tolist(),[0,1,4,5]);self.assertTrue(torch.allclose(t['boxes'][:,0],torch.full((4,),.5),atol=.002))
        y=subset_targets(t,torch.tensor([4,5]),torch.empty(6,1,2,2));self.assertEqual(y['batch_idx'].tolist(),[0,1]);self.assertEqual(len(y['img']),2)
    def test_no_labels_and_invalid_tir(self):
        b,e=self.batch();b['tir_batch_idx']=torch.empty(0,dtype=torch.long);b['tir_bboxes']=torch.empty(0,4);b['tir_cls']=torch.empty(0,1)
        e['roi'].tir_valid[-1]=.2;t=local_targets(b,e,'tir');self.assertEqual(t['valid'].tolist(),[False,False,False,False,True,False]);self.assertEqual(tuple(t['boxes'].shape),(0,4))
    def test_independent_mono_head(self):
        class FakeDetect(nn.Module):
            def __init__(self):
                super().__init__();self.cv2=nn.ModuleList([nn.Sequential(nn.Conv2d(c,4,1)) for c in (4,8,8)])
                self.stride=torch.tensor([8.,16.,32.])
            def forward(self,features):return [m(x) for m,x in zip(self.cv2,features)]
        base=FakeDetect();mono=MonoHead(base,(8,8,8));self.assertFalse(set(map(id,base.parameters())) & set(map(id,mono.parameters())))
        feat=[torch.randn(2,8,h,h,requires_grad=True) for h in (8,4,2)]
        raw=mono(feat);sum(x.square().mean() for x in raw).backward()
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in feat));self.assertTrue(all(p.grad is None for p in base.parameters()))
        mono.double();self.assertEqual(mono.detect.stride.dtype,torch.float64)
    def test_recovery_matching(self):
        gt={'bboxes':torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]]),'cls':torch.zeros(2)}
        p={'bboxes':torch.tensor([[0.,0.,10.,10.],[0.,0.,10.,10.],[20.,20.,30.,30.]]),'cls':torch.zeros(3),'conf':torch.tensor([.9,.8,.1])}
        self.assertEqual(hit_mask(p,gt).tolist(),[True,False])
if __name__=='__main__':unittest.main()
