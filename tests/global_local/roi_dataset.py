"""Offline local RGB-T ROI dataset; compatible with v5 RGBTDetectionModel.loss.

Images and paired labels are emitted by export_global_local_rois.py; the
original UAV-CB split folders are untouched.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RGBTLocalROIDataset(Dataset):
    def __init__(self, roi_root: str | Path, split: str, augment: bool = False,
                 fliplr: float = 0.5, max_images: int = 0):
        self.root = Path(roi_root).expanduser().resolve()
        self.split = str(split)
        manifest = self.root / f'{split}.jsonl'
        if not manifest.is_file():
            raise FileNotFoundError(f'No ROI manifest {manifest}. Run export_global_local_rois.py')
        with manifest.open('r',encoding='utf-8') as fh:
            self.records = [json.loads(s) for s in fh if s.strip()]
        if max_images:
            self.records = self.records[:int(max_images)]
        if not self.records:
            raise RuntimeError(f'ROI split is empty: {manifest}')
        self.augment = bool(augment)
        self.fliplr = float(fliplr)

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        item=self.records[index]
        rgb_path = self.root / item['rgb_path']
        tir_path = self.root / item['tir_path']
        rgb=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        tir=cv2.imread(str(tir_path),cv2.IMREAD_GRAYSCALE)
        if rgb is None or tir is None:
            raise FileNotFoundError(f'Failed reading {rgb_path} or {tir_path}')
        rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
        if rgb.shape[:2]!=tir.shape[:2]:
            raise ValueError(f'Unpaired ROI sizes: {rgb.shape} / {tir.shape}')
        boxes=np.array(item.get('boxes_xywh', []),dtype=np.float32).reshape(-1,4)
        if self.augment and random.random()<self.fliplr:
            rgb=np.fliplr(rgb)
            tir=np.fliplr(tir)
            boxes=boxes.copy()
            boxes[:,0]=1.0-boxes[:,0]
        hh,ww=rgb.shape[:2]
        rgb=np.ascontiguousarray(rgb.transpose(2,0,1))
        tir=np.ascontiguousarray(np.repeat(tir[...,None],3,axis=2).transpose(2,0,1))
        rect=torch.tensor(boxes,dtype=torch.float32)
        cls=torch.zeros((len(boxes),1),dtype=torch.float32)
        ratio_pad={'ratio':1.0,'pad':(0,0)}
        return {
            'rgb_img':torch.from_numpy(rgb),
            'tir_img':torch.from_numpy(tir),
            'rgb_semantic_img':torch.from_numpy(rgb.copy()),
            'rgb_cls':cls,'rgb_bboxes':rect,'rgb_semantic_bboxes':rect.clone(),
            'tir_cls':cls.clone(),'tir_bboxes':rect.clone(),
            'rgb_path':str(rgb_path),'tir_path':str(tir_path),
            'rgb_ratio_pad':ratio_pad,'rgb_semantic_ratio_pad':ratio_pad,
            'tir_ratio_pad':ratio_pad,'rgb_ori_shape':(hh,ww),'tir_ori_shape':(hh,ww),
            'pair_key':item['id'],
        }

    @staticmethod
    def collate_fn(samples):
        batch={k:torch.stack([s[k] for s in samples]) for k in
               ('rgb_img','rgb_semantic_img','tir_img')}
        for tag in ('rgb','tir'):
            boxes=[s[f'{tag}_bboxes'] for s in samples]
            clss=[s[f'{tag}_cls'] for s in samples]
            batch[f'{tag}_bboxes']=torch.cat(boxes,dim=0)
            batch[f'{tag}_cls']=torch.cat(clss,dim=0)
            batch[f'{tag}_batch_idx']=torch.cat([torch.full((len(b),),i,dtype=torch.long) for i,b in enumerate(boxes)])
        batch['rgb_semantic_bboxes']=batch['rgb_bboxes'].clone()
        for k in ('rgb_path','tir_path','rgb_ratio_pad','rgb_semantic_ratio_pad','tir_ratio_pad',
                  'rgb_ori_shape','tir_ori_shape','pair_key'):
            batch[k]=[s[k] for s in samples]
        return batch
