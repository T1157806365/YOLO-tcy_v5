"""CPU unit tests for GPR-MS KD; no Ultralytics or user data required."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gmls_roi_hybrid.distillation import (
    kd_weight, matched_teacher_targets, p3_roi_attention, p4_fg_bg_contrast,
    p5_semantic_relation, _pair_iou,
)


def test_schedule():
    assert kd_weight(0)==0 and kd_weight(9)==0
    assert abs(kd_weight(19)-.01)<1e-12
    assert abs(kd_weight(20)-.01)<1e-12
    assert abs(kd_weight(59)-.01)<1e-12
    assert abs(kd_weight(79)-0)<1e-12
    assert kd_weight(80)==0 and kd_weight(99)==0


def test_match_rgb_tir_coordinates_independent():
    dev='cpu'
    # first image has one true target, second has one unmatched target.
    gt=torch.tensor([[.50,.50,.2,.2],[.4,.4,.1,.1]],device=dev)
    idx=torch.tensor([0,1]);cls=torch.tensor([0,0]);images=torch.zeros(2,3,640,640)
    batch={'rgb_semantic_bboxes':gt, 'rgb_batch_idx':idx,'rgb_cls':cls,
           'rgb_semantic_img':images,'tir_bboxes':gt,'tir_batch_idx':idx,
           'tir_cls':cls,'tir_img':images}
    pred=[{'bboxes':torch.tensor([[256.,256.,384.,384.]]),'conf':torch.tensor([.9]),'cls':torch.tensor([0])},
          {'bboxes':torch.tensor([[50.,50.,100.,100.]]),'conf':torch.tensor([.8]),'cls':torch.tensor([0])}]
    for mod in ('rgb','tir'):
        boxes,q,nvalid,ntotal=matched_teacher_targets(batch,pred,mod)
        assert ntotal==2 and nvalid==1 and len(boxes[0])==1 and len(boxes[1])==0
        assert abs(float(q[0])-.9)<1e-5 and q[1]==0


def test_p3_teacher_constant_zero_kd_and_gradient():
    # A perfect match should have ~zero KL; mismatch should backpropagate.
    teacher=torch.randn(2,12,80,80)
    student=teacher.detach().clone().requires_grad_(True)
    box=[torch.tensor([[.4,.4,.1,.1]]),torch.tensor([[.5,.5,.03,.03]])]
    q=torch.tensor([.7,.9])
    kl=p3_roi_attention(student,teacher,box,q)
    assert abs(float(kl))<1e-4,kl
    altered=(teacher*.2+torch.randn_like(teacher)*.8).requires_grad_(True)
    kl2=p3_roi_attention(altered,teacher,box,q)
    kl2.backward()
    assert torch.isfinite(kl2) and altered.grad is not None and altered.grad.abs().sum()>0


def test_p4_p5_losses_gradients():
    a=torch.randn(2,24,40,40,requires_grad=True)
    t=torch.randn_like(a)
    box=[torch.tensor([[.5,.5,.08,.08]]),torch.tensor([[.2,.6,.1,.15]])]
    q=torch.tensor([.9,.4])
    p4=p4_fg_bg_contrast(a,t,box,q)
    p5=p5_semantic_relation(a,t,box,q)
    assert torch.isfinite(p4+p5)
    (p4+p5).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum()>0


def test_no_valid_gt_connected_zero():
    a=torch.randn(2,16,20,20,requires_grad=True)
    t=torch.randn_like(a)
    boxes=[a.new_zeros((0,4)),a.new_zeros((0,4))]
    q=torch.zeros(2)
    loss=p3_roi_attention(a,t,boxes,q)+p4_fg_bg_contrast(a,t,boxes,q)+p5_semantic_relation(a,t,boxes,q)
    loss.backward()
    assert float(loss)==0 and a.grad is not None and torch.isfinite(a.grad).all()


def test_iou_edges():
    x=torch.tensor([[0.,0.,10.,10.]])
    assert torch.allclose(_pair_iou(x,x),torch.ones(1,1))
    assert float(_pair_iou(x,x+20))==0.0
