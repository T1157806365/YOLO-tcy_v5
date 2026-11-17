"""CPU-only geometry / label / matching tests; no ultralytics required."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_local_scratch.geometry import (
    high_to_semantic, semantic_to_high,sample_native_hr_and_tir,
    local_gt_from_high, local_box_to_semantic,
)
from global_local_scratch.fusion import safe_replace, merge_local_predictions


def test_geometry_and_labels():
    high={'ratio':1.0,'pad':(0,0)}
    sem={'ratio':1/3,'pad':(0,0)}
    xy=torch.tensor([[230.,145.],[150.,440.]])
    assert torch.allclose(high_to_semantic(semantic_to_high(xy,sem,high),high,sem),xy,atol=1e-4)
    rgb=torch.zeros(3,384,384)
    tir=torch.full((3,128,128),0.3)
    high={'ratio':1.0,'pad':(0,0)};sem={'ratio':1/3,'pad':(0,0)}
    r,t,geo=sample_native_hr_and_tir(
      rgb,tir,torch.tensor([60.,60.]),(128,128),high,sem,
      torch.ones(2),torch.zeros(2),roi_size=96)
    assert r.shape==(3,96,96) and t.shape==(3,96,96)
    assert abs(geo.tir_valid_fraction-1.)<1e-6
    assert torch.allclose(t.mean(),torch.tensor(.3),atol=1e-5)
    b=torch.tensor([[180/384,180/384,18/384,18/384]])
    local=local_gt_from_high(b,geo.x0,geo.y0,96,384,384)
    assert local.shape==(1,4)
    lo=local_box_to_semantic(torch.tensor([[180-geo.x0-9,180-geo.y0-9,
                                          180-geo.x0+9,180-geo.y0+9]],dtype=torch.float32),
                             geo,high,sem,(128,128))
    assert torch.allclose(lo,torch.tensor([[57.,57.,63.,63.]]),atol=1e-4)


def test_fusion():
    gb={'bboxes':torch.tensor([[20.,20.,40.,40.]]),
        'conf':torch.tensor([.8]),'cls':torch.tensor([0.])}
    lp={'bboxes':torch.tensor([[21.,21.,39.,39.]]),
        'conf':torch.tensor([.7]),'cls':torch.tensor([0.])}
    out,st=safe_replace(gb,lp,min_local_conf=.2)
    assert st['replaced']==1 and torch.equal(out['conf'],gb['conf'])
    assert len(merge_local_predictions([lp,lp])['bboxes'])==1

if __name__=='__main__':
    test_geometry_and_labels();test_fusion();print('[PASS] ROI geometry, TIR grid, boxes, safe merge')
