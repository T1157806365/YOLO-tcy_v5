"""CPU-only unit tests; no YOLO repo needed.
Run: python tests/test_core.py
"""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ag_lhfe.local_hr import LocalHighResFeatureExtractor

class LocalHRTests(unittest.TestCase):
    def make_model(self, mode):
        return LocalHighResFeatureExtractor(
            channels=32, proposal_mode=mode, topk=8,
            nms_radius=1, patch_size=16,crop_sizes_high=(16,32),
            latent_dim=16, strength_init=.05, strength_max=.2,
            positive_radius=2.5,
        )

    def test_forward_gradient_and_gt_recall(self):
        torch.manual_seed(1)
        for mode in ('rgb','rgb_align'):
            m=self.make_model(mode).train()
            high=torch.rand(2,3,96,96)
            rgb=torch.randn(2,32,8,8,requires_grad=True)
            tir=torch.randn(2,32,8,8)
            sr=torch.rand(2,1,8,8)
            st=torch.rand(2,1,8,8)
            joint=(sr*st).sqrt()
            output,state=m(high,rgb,tir,sr,st,joint)
            self.assertEqual(output.shape,rgb.shape)
            self.assertEqual(state['coords'].shape,(2,8,2))
            batch={'rgb_semantic_bboxes':torch.tensor([[.4,.3,.08,.06],[.8,.6,.05,.05]]),
                   'rgb_batch_idx':torch.tensor([[0],[1]])}
            aux,items=m.auxiliary_loss(state,batch)
            total=output.square().mean()+aux
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertTrue(rgb.grad is not None and rgb.grad.abs().sum()>0)
            self.assertTrue(m.rgb_encoder.body[0][0].weight.grad is not None)
            self.assertTrue(m.strength_logit.grad is not None)
            self.assertTrue(0<=float(items['lhfe_proposal_gt_recall'])<=1)

    def test_injection_off_changes_output_but_same_proposals(self):
        torch.manual_seed(2)
        m=self.make_model('rgb_align').eval()
        high=torch.rand(1,3,96,96);rgb=torch.randn(1,32,8,8);tir=torch.randn_like(rgb)
        sr=torch.rand(1,1,8,8);j=torch.rand_like(sr)
        with torch.no_grad():
            on,state1=m(high,rgb,tir,sr,j,j)
            m.injection_enabled=False
            off,state2=m(high,rgb,tir,sr,j,j)
        self.assertEqual(float((off-rgb).abs().max()),0.)
        self.assertTrue(float((on-rgb).abs().max())>0)
        self.assertTrue(torch.equal(state1['coords'],state2['coords']))

    def test_crop_center_maps_to_high_rgb_correctly(self):
        m=self.make_model('rgb')
        # RGB image has X ramp, center P3 (x=3,y=2) maps to high pixel 41.5.
        high=torch.arange(96.).view(1,1,1,96).expand(1,3,96,96)/96.
        pt=torch.tensor([[[3.,2.]]])
        crop=m._crop_rgb(high,pt,(8,8),16.)
        self.assertEqual(crop.shape,(1,3,16,16))
        self.assertAlmostEqual(float(crop[0,0].mean()), (3.5/8 - 0.5/96),places=3)

if __name__=='__main__': unittest.main(verbosity=2)
