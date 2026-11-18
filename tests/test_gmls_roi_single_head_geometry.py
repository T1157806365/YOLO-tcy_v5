"""CPU geometry, differentiability and correct ROI-label transform tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gmls_roi_single_head.geometry import (
    semantic_to_high_norm,sample_roi_pair,local_labels_from_high,
    boxes_roi_to_semantic,roi_gt_coverage,
)


def test_identical_crop_and_grad():
    torch.manual_seed(17)
    b=2; high=torch.rand(b,3,60,60);tir=torch.rand(b,3,20,20)
    centers=torch.tensor([[.5,.45],[.72,.62]],requires_grad=True)
    scale=torch.ones(b,2,requires_grad=True)
    shift=torch.zeros(b,2,requires_grad=True)
    hm=[{'ratio':3.,'pad':(0,0)} for _ in range(b)]
    sm=[{'ratio':1.,'pad':(0,0)} for _ in range(b)]
    rgb,thermal,roi=sample_roi_pair(high,tir,centers,scale,shift,hm,sm,(20,20),size=20,
                                    geometry_grad_scale=.1,center_grad_scale=.1)
    assert rgb.shape==(2,3,20,20) and thermal.shape==(2,3,20,20)
    assert torch.isfinite(rgb).all() and torch.isfinite(thermal).all()
    assert (roi.tir_valid>0).all()
    (rgb.square().mean()+thermal.square().mean()).backward()
    assert centers.grad is not None and torch.isfinite(centers.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    assert shift.grad is not None and torch.isfinite(shift.grad).all()


def test_exact_letterbox_and_box_roundtrip():
    high=torch.zeros(1,3,60,60);tir=torch.zeros(1,3,20,20)
    hm=[{'ratio':3.,'pad':(0,0)}];sm=[{'ratio':1.,'pad':(0,0)}]
    center=torch.tensor([[.5,.5]],requires_grad=True)
    rgb,thermal,geo=sample_roi_pair(high,tir,center,torch.ones(1,2),torch.zeros(1,2),hm,sm,(20,20),20)
    assert torch.allclose(geo.xy0,torch.tensor([[18.5,18.5]]),atol=1e-4)
    # Center high = 28.5 px, box around 28.5 px, crop origin 18.5 px.
    # Pixel [24,24] .. [32,32] => visible, normalized center 10/20.
    batch={
        'rgb_img':high, 'tir_img':tir,'rgb_semantic_img':torch.zeros(1,3,20,20),
        'rgb_bboxes':torch.tensor([[28.5/60,28.5/60,8/60,8/60]]),
        'rgb_batch_idx':torch.tensor([0]),'rgb_cls':torch.zeros(1,1),
        'rgb_ratio_pad':hm,'rgb_semantic_ratio_pad':sm,
    }
    labels,positive=local_labels_from_high(batch,geo,.6)
    assert positive==1
    expected=torch.tensor([[.5,.5,.4,.4]])
    assert torch.allclose(labels['rgb_bboxes'],expected,atol=1e-5)
    # Pred ROI bbox [6,6,14,14] mapped to 20x20 semantic coordinates.
    local_xyxy=torch.tensor([[6.,6.,14.,14.]])
    sem_box=boxes_roi_to_semantic(local_xyxy,geo,batch,0)
    assert torch.allclose(sem_box,torch.tensor([[24.5/3,24.5/3,32.5/3,32.5/3]]),atol=1e-5)
    hit,n=roi_gt_coverage(batch,geo)
    assert (hit,n)==(1,1)


def test_padding_aware_mapping():
    h=[{'ratio':2.5,'pad':(8,20)}]
    s=[{'ratio':.833333333333,'pad':(3,7)}]
    pos=torch.tensor([[.33,.62]])
    x=semantic_to_high_norm(pos,h,s,(1920,1920),(640,640))
    expect=(pos*639-torch.tensor([[3.,7.]]))/.833333333333*2.5+torch.tensor([[8.,20.]])
    assert torch.allclose(x,expect,atol=1e-4)


if __name__=='__main__':
    test_identical_crop_and_grad();test_exact_letterbox_and_box_roundtrip();test_padding_aware_mapping()
    print('[ALL PASS] ROI geometry and gradients, GT box roundtrip, letterbox padding')
