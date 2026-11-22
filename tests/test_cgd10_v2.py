"""CPU checks independent of YOLO teacher checkpoints or user CUDA hardware."""
import unittest
import torch
from cgd10_v2.losses import (
    response_from_p3,response_moments,route_offset_response_kd,
    final_feature_warp_kd,warp_feature_response)

class CGD10FeatureGeometryV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.rb=torch.tensor([[.47,.51,.04,.04],[.35,.36,.05,.04]])
        self.tb=torch.tensor([[.50,.53,.05,.05],[.38,.37,.06,.05]])
        self.fr=torch.randn(2,32,80,80)
        self.ft=torch.randn(2,32,80,80)
        self.rr=response_from_p3(self.fr,self.rb)
        self.tr=response_from_p3(self.ft,self.tb)

    def test_actual_feature_not_gaussian(self):
        self.assertEqual(self.rr['heatmap'].shape,(2,1,80,80))
        self.assertTrue(torch.allclose(self.rr['heatmap'].sum((2,3)),torch.ones((2,1)),atol=1e-5))
        self.assertTrue(torch.isfinite(self.rr['center']).all())
        # P3 energy perturbation changes the teacher response for the same GT box.
        altered=self.fr.clone();altered[0,:,40,38]*=50
        changed=response_from_p3(altered,self.rb)
        self.assertFalse(torch.allclose(changed['heatmap'][0],self.rr['heatmap'][0]))

    def test_route_and_offset_have_separate_gradients(self):
        logits=torch.tensor([[.2,-.1],[.1,.2]],requires_grad=True)
        goff=torch.tensor([[1.9,1.1],[1.0,.2]],requires_grad=True)
        doff=torch.tensor([[2.8,1.6],[3.,1.9]],requires_grad=True)
        route,offset,stat=route_offset_response_kd(
            logits,goff,doff,self.rb,self.tb,self.rr['center'],self.tr['center'],
            torch.tensor([.9,.8]),route_gt_margin=100)
        grads=torch.autograd.grad(route+offset,(logits,goff,doff))
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())
            self.assertGreater(float(g.abs().sum()),0)
        self.assertGreater(float(stat['route_eligible']),0)

    def test_final_geometry_grad(self):
        scale=torch.tensor([[1.1,1.05],[.95,1.2]],requires_grad=True)
        trans=torch.tensor([[.01,.03],[.08,-.01]],requires_grad=True)
        loss,stats=final_feature_warp_kd(scale,trans,self.rr,self.tr,self.rb,self.tb,
                                           torch.tensor([.9,.8]),overlap_weight=.05)
        g=torch.autograd.grad(loss,(scale,trans))
        self.assertTrue(all(torch.isfinite(x).all() for x in g))
        self.assertTrue(all(float(x.abs().sum())>0 for x in g))
        self.assertTrue(torch.isfinite(stats['warp_response_overlap']))

    def test_native_rgb_tir_batch_keys(self):
        from cgd10_v2.teachers import choose_teacher_boxes
        batch={
            'rgb_semantic_img':torch.zeros(2,3,640,640),
            'tir_img':torch.zeros(2,3,640,640),
            'rgb_batch_idx':torch.tensor([0,1]),
            'tir_batch_idx':torch.tensor([0,1]),
            'rgb_cls':torch.tensor([[0.],[0.]]),
            'tir_cls':torch.tensor([[0.],[0.]]),
            'rgb_semantic_bboxes':self.rb,
            'tir_bboxes':self.tb,
        }
        pred=[]
        for box in self.rb:
            xyxy=torch.cat((box[:2]-box[2:]/2,box[:2]+box[2:]/2))*640
            pred.append({'bboxes':xyxy[None],'conf':torch.tensor([.98]),
                         'cls':torch.tensor([0.])})
        rgb=choose_teacher_boxes(batch,pred,'rgb')
        self.assertTrue(rgb['valid'].all())
        pred=[]
        for box in self.tb:
            xyxy=torch.cat((box[:2]-box[2:]/2,box[:2]+box[2:]/2))*640
            pred.append({'bboxes':xyxy[None],'conf':torch.tensor([.98]),
                         'cls':torch.tensor([0.])})
        tir=choose_teacher_boxes(batch,pred,'tir')
        self.assertTrue(tir['valid'].all())

    def test_reliability_zero_safe(self):
        scale=torch.ones(2,2,requires_grad=True)
        trans=torch.zeros(2,2,requires_grad=True)
        logits=torch.randn(2,2,requires_grad=True)
        a=torch.randn(2,2,requires_grad=True)
        b=torch.randn(2,2,requires_grad=True)
        z=torch.zeros(2)
        r,o,_=route_offset_response_kd(logits,a,b,self.rb,self.tb,
                                         self.rr['center'],self.tr['center'],z)
        w,_=final_feature_warp_kd(scale,trans,self.rr,self.tr,self.rb,self.tb,z)
        loss=r+o+w
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for t in (scale,trans,logits,a,b):
            self.assertTrue(torch.isfinite(t.grad).all())

if __name__=='__main__':unittest.main()
