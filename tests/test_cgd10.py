"""CPU mathematical tests. Run: python -m unittest discover -s tests -p test_cgd10.py"""
import unittest
import torch
from cgd10.losses import gaussian_response,warp_teacher_response,route_teacher_kd,final_warp_kd


class CGD10Tests(unittest.TestCase):
    def test_geometry_identity_and_backprop(self):
        a=torch.tensor([[.5,.45,.07,.06]])
        z=gaussian_response(a,48,48)
        s=torch.tensor([[1.04,.97]],requires_grad=True)
        t=torch.tensor([[-.03,.015]],requires_grad=True)
        w=warp_teacher_response(z,s,t)
        self.assertEqual(tuple(w.shape),(1,1,48,48))
        loss,diagnostics=final_warp_kd(s,t,a,a,torch.tensor([.9]))
        g=torch.autograd.grad(loss,(s,t),allow_unused=False)
        self.assertTrue(all(torch.isfinite(x).all() for x in g))
        self.assertGreater(float(g[0].abs().sum()+g[1].abs().sum()),0.)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(diagnostics['warp_soft_overlap']))

    def test_routing_kd_gradient(self):
        logits=torch.tensor([[.2,-.1],[.1,.2]],requires_grad=True)
        goff=torch.tensor([[2.,1.],[1.,0.]])
        doff=torch.tensor([[4.,1.],[3.,2.]])
        gt_rgb=torch.tensor([[.5,.5,.05,.05],[.45,.45,.05,.05]])
        gt_tir=torch.tensor([[.53,.51,.05,.05],[.49,.46,.05,.05]])
        r,meta=route_teacher_kd(logits,goff,doff,gt_rgb,gt_tir,
                     gt_rgb,gt_tir,torch.tensor([.8,.9]),gt_margin=100)
        r.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        self.assertGreater(float(meta['route_eligible']),0.)

    def test_zero_weight_no_nan(self):
        s=torch.ones((1,2),requires_grad=True)
        t=torch.zeros((1,2),requires_grad=True)
        box=torch.tensor([[.5,.5,.04,.04]])
        loss,_=final_warp_kd(s,t,box,box,torch.zeros(1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertTrue(torch.isfinite(t.grad).all())

if __name__=='__main__':unittest.main()
