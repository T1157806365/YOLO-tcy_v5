"""CPU-only tests, no ultralytics dependency."""
from __future__ import annotations
import json, tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import cv2
import numpy as np
from global_local.geometry import (high_to_semantic, semantic_to_high, sample_native_hr_and_tir,
                                   local_gt_from_high, local_box_to_semantic)
from global_local.fusion import pairwise_iou, safe_replace, merge_local_predictions
from global_local.roi_dataset import RGBTLocalROIDataset


def test_geometry():
    high={'ratio':1.0,'pad':(0,0)};sem={'ratio':1/3,'pad':(0,0)}
    q=torch.tensor([[230.,145.],[150.,440.]])
    assert torch.allclose(high_to_semantic(semantic_to_high(q,sem,high),high,sem),q,atol=1e-4)
    rgb=torch.zeros((3,1920,1920))
    tir=torch.zeros((3,640,640));tir[:,:,:]=0.3
    r,t,g=sample_native_hr_and_tir(rgb,tir,torch.tensor([300.,300.]),(640,640),high,sem,
                                    torch.ones(2),torch.zeros(2),roi_size=96)
    assert r.shape==(3,96,96) and t.shape==(3,96,96)
    assert (g.x0,g.y0)==(852,852)
    assert abs(g.tir_valid_fraction-1.0)<1e-6
    assert torch.allclose(t.mean(),torch.tensor(.3),atol=1e-5)
    gt=torch.tensor([[900/1920,900/1920,18/1920,18/1920]])
    b=local_gt_from_high(gt,g.x0,g.y0,96,1920,1920)
    assert b.shape==(1,4) and torch.allclose(b[0,:2],torch.tensor([.5,.5]),atol=1e-5)
    bb=torch.tensor([[39.,39.,57.,57.]])
    recovered=local_box_to_semantic(bb,g,high,sem,(640,640))
    assert torch.allclose(recovered,torch.tensor([[297.,297.,303.,303.]]),atol=1e-3)
    print('[PASS] letterbox roundtrip, native crop, TIR warp, local GT, global box restore')


def test_fusion():
    global_p={'bboxes':torch.tensor([[20.,20.,40.,40.]]),'conf':torch.tensor([.8]),'cls':torch.tensor([0.])}
    local={'bboxes':torch.tensor([[21.,21.,39.,39.]]),'conf':torch.tensor([.7]),'cls':torch.tensor([0.])}
    combined,change=safe_replace(global_p,local,min_local_conf=.2)
    assert change['replaced']==1 and change['added']==0
    assert torch.equal(combined['bboxes'],local['bboxes'])
    assert torch.equal(combined['conf'],global_p['conf'])
    assert torch.equal(global_p['bboxes'],torch.tensor([[20.,20.,40.,40.]]))
    merged=merge_local_predictions([local,local],nms_iou=.5)
    assert len(merged['bboxes'])==1
    print('[PASS] cautious replacement preserves global confidence; ROI dedup')


def test_dataset():
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        (d/'images/train/RGB').mkdir(parents=True)
        (d/'images/train/TIR').mkdir(parents=True)
        cv2.imwrite(str(d/'images/train/RGB/a.jpg'),np.zeros((96,96,3),dtype=np.uint8))
        cv2.imwrite(str(d/'images/train/TIR/a.jpg'),np.zeros((96,96),dtype=np.uint8))
        rows=[{'id':'a','rgb_path':'images/train/RGB/a.jpg','tir_path':'images/train/TIR/a.jpg',
                'boxes_xywh':[[.5,.5,.2,.2]]},
              {'id':'a','rgb_path':'images/train/RGB/a.jpg','tir_path':'images/train/TIR/a.jpg',
                'boxes_xywh':[]}]
        with (d/'train.jsonl').open('w') as fh:
            for row in rows:fh.write(json.dumps(row)+'\n')
        ds=RGBTLocalROIDataset(d,'train',augment=False)
        b=RGBTLocalROIDataset.collate_fn([ds[0],ds[1]])
        assert b['rgb_img'].shape==(2,3,96,96)
        assert b['rgb_bboxes'].shape==(1,4)
        assert b['rgb_batch_idx'].tolist()==[0]
        assert b['tir_img'].shape==(2,3,96,96)
    print('[PASS] ROI manifest dataset + negative crops + v5 batch contract')

if __name__=='__main__':
    test_geometry();test_fusion();test_dataset()
