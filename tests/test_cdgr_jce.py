import unittest
import torch
from torch import nn
from cdgr_jce.fusion import JointEncoder,JointFeedback,group_concat
from cdgr_jce.progressive import run_features
from cdgr_jce.robust import subset_labels,freeze_bn_statistics

torch.set_num_threads(1)

def layer_input(m,x,outputs):
    return x if m.f==-1 else outputs[m.f] if isinstance(m.f,int) else [x if j==-1 else outputs[j] for j in m.f]

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.ModuleList(nn.Conv2d(8,8,1,bias=False) for _ in range(11));self.tir_backbone=nn.ModuleList(nn.Conv2d(8,8,1,bias=False) for _ in range(11));self.backbone_len=11
        for m in [*self.model,*self.tir_backbone]:m.f=-1;nn.init.dirac_(m.weight)
        self.fusion_indices=[4,6,10];self.jce_blocks=nn.ModuleDict({str(i):JointEncoder(8,4,2) for i in self.fusion_indices});self.jce_feedback=nn.ModuleDict({str(i):JointFeedback(8,2) for i in (4,6)});self.jce_intervention='normal';self.jce_weak_rgb=False

class Tests(unittest.TestCase):
    def test_group_layout(self):
        a=torch.arange(4.).view(1,4,1,1);b=a+10
        self.assertEqual(group_concat((a,b),2).flatten().tolist(),[0,1,10,11,2,3,12,13])
    def test_shapes_and_gradients(self):
        for h in (80,40,20):
            m=JointEncoder();m.collect_stats=True;r=torch.randn(1,512,h,h,requires_grad=True);t=torch.randn_like(r,requires_grad=True)
            z=m(r,t);self.assertEqual(z.shape,r.shape);z.square().mean().backward()
            for p in [r,t,*m.parameters()]:
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
            self.assertGreater(float(m.last_stats['delta_over_add']),0)
    def test_close_to_add_not_equal(self):
        m=JointEncoder();r=torch.randn(1,512,8,8);t=torch.randn_like(r);z=m(r,t)
        ratio=(z-r-t).abs().mean()/(r+t).abs().mean();self.assertGreater(float(ratio),.005);self.assertLess(float(ratio),.2)
        m.intervention='add';torch.testing.assert_close(m(r,t),r+t,atol=0,rtol=0)
    def test_feedback_and_native_indices(self):
        det=Toy();r=torch.randn(1,8,8,8);t=torch.randn_like(r)
        a=run_features(det,r,t,layer_input);det.jce_intervention='no_feedback';b=run_features(det,r,t,layer_input)
        torch.testing.assert_close(a['fusion_features'][4],b['fusion_features'][4]);self.assertGreater(float((a['rgb_backbone'][10]-b['rgb_backbone'][10]).abs().sum()),0);self.assertGreater(float((a['tir_backbone'][10]-b['tir_backbone'][10]).abs().sum()),0)
        det.jce_intervention='add';c=run_features(det,r,t,layer_input)
        for i in det.fusion_indices:torch.testing.assert_close(c['fusion_features'][i],r+t)
    def test_modal_intervention(self):
        det=Toy();r=torch.randn(1,8,8,8);t=torch.randn_like(r)
        det.jce_intervention='zero_tir';a=run_features(det,r,t,layer_input);b=run_features(det,r,t*10,layer_input)
        for i in det.fusion_indices:torch.testing.assert_close(a['fusion_features'][i],b['fusion_features'][i])
        det.jce_intervention='rgb_only';a=run_features(det,r,t,layer_input)
        for i in det.fusion_indices:torch.testing.assert_close(a['fusion_features'][i],r)
    def test_weak_rgb_auxiliary(self):
        det=Toy();r=torch.randn(2,8,8,8);t=torch.randn_like(r);a=run_features(det,r,t,layer_input);det.jce_weak_rgb=True;b=run_features(det,r,t,layer_input);self.assertGreater(float((a['fusion_features'][4]-b['fusion_features'][4]).abs().sum()),0)
        b['fusion_features'][10].square().mean().backward()
        for p in det.jce_feedback.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
    def test_subset_labels(self):
        x=torch.randn(4,3,8,8,requires_grad=True);y={'rgb_img':x,'tir_img':x*2,'rgb_cls':torch.tensor([[0.],[1.],[0.]]),'rgb_bboxes':torch.rand(3,4),'rgb_batch_idx':torch.tensor([0,2,3])}
        sub=subset_labels(y,torch.tensor([2,3]));self.assertEqual(sub['rgb_batch_idx'].tolist(),[0,1]);self.assertEqual(len(sub['rgb_bboxes']),2);self.assertFalse(sub['rgb_img'].requires_grad);self.assertFalse(sub['tir_img'].requires_grad)
    def test_bn_policy(self):
        net=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4));net.train();x=torch.randn(2,3,8,8);a=net(x);mean=net[1].running_mean.clone()
        with freeze_bn_statistics(net):
            self.assertTrue(net.training);self.assertFalse(net[1].training);b=net(x*2)
        self.assertTrue(net[1].training);torch.testing.assert_close(net[1].running_mean,mean);(a.square().mean()+b.square().mean()).backward();self.assertIsNotNone(net[1].weight.grad)
    def test_optimizer(self):
        m=JointEncoder(8,4,2);opt=torch.optim.SGD(m.parameters(),lr=.01);r=torch.randn(1,8,8,8);t=torch.randn_like(r);old=m.r[0].weight.detach().clone();m(r,t).square().mean().backward();opt.step();self.assertGreater(float((m.r[0].weight-old).abs().sum()),0)

if __name__=='__main__':unittest.main()
