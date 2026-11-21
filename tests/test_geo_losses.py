"""Independent geometry-only tests; run with python -m pytest tests/test_geo_losses.py -q."""
import torch
import pytest
from gmls_roi_hybrid.geo_losses import final_center_loss, final_box_geometry_loss, _iou_giou_xywh


def batch_one():
    return {
        'rgb_semantic_img': torch.zeros(1, 3, 640, 640),
        'rgb_semantic_bboxes': torch.tensor([[.45,.60,.10,.12]]),
        'tir_bboxes': torch.tensor([[.45,.60,.10,.12]]),
        'rgb_batch_idx': torch.tensor([0]),
        'tir_batch_idx': torch.tensor([0]),
    }


def test_identity_geometric_alignment():
    b=batch_one()
    scale=torch.ones(1,2,requires_grad=True)
    shift=torch.zeros(1,2,requires_grad=True)
    info={'sampling_scale':scale,'sampling_translation':shift}
    c, n, _=final_center_loss(b,info)
    s,g,n2,iou,e=final_box_geometry_loss(b,info)
    assert n==1 and n2==1 and iou.item()==pytest.approx(1.,abs=1e-6)
    assert torch.isfinite(s+g+c) and s.item()<1e-6 and g.item()<1e-6
    assert e.item()<1e-6


def test_geometry_gradients():
    b=batch_one()
    scale=torch.tensor([[1.2,.9]],requires_grad=True)
    shift=torch.tensor([[.1,-.03]],requires_grad=True)
    s,g,n,iou,e=final_box_geometry_loss(b,{'sampling_scale':scale,'sampling_translation':shift})
    loss=.05*s+.1*g
    loss.backward()
    assert n==1
    assert scale.grad is not None and torch.isfinite(scale.grad).all() and scale.grad.abs().sum()>0
    assert shift.grad is not None and torch.isfinite(shift.grad).all() and shift.grad.abs().sum()>0


def test_empty_gt():
    b=batch_one()
    for name in ('rgb_semantic_bboxes','tir_bboxes'):
        b[name]=torch.zeros(0,4)
    for name in ('rgb_batch_idx','tir_batch_idx'):
        b[name]=torch.zeros(0,dtype=torch.long)
    scale=torch.ones(1,2,requires_grad=True)
    shift=torch.zeros(1,2,requires_grad=True)
    s,g,n,iou,e=final_box_geometry_loss(b,{'sampling_scale':scale,'sampling_translation':shift})
    assert n==0 and s.item()==g.item()==0
    (s+g).backward()
    assert torch.isfinite(scale.grad).all() and torch.isfinite(shift.grad).all()


def test_nonoverlap_giou_finite():
    a=torch.tensor([[.1,.1,.05,.05]],requires_grad=True)
    b=torch.tensor([[.9,.9,.05,.05]])
    i,g=_iou_giou_xywh(a,b)
    assert i.item()==0. and torch.isfinite(g).all()
    (1-g.mean()).backward()
    assert torch.isfinite(a.grad).all()
