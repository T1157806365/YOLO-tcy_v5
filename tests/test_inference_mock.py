"""CPU synthetic global→ROI inference test without Ultralytics installed.

Uses only fixed dummy detections/GTGA geometry; validates end-to-end crop
plumbing and GT-free fallback. Does not replace real-model CUDA smoke tests.
"""
from __future__ import annotations
import sys,types
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
engine=types.ModuleType('engine')
evaluator=types.ModuleType('engine.evaluator')
evaluator.postprocess_predictions=lambda pred,**kwargs:pred
engine.evaluator=evaluator
sys.modules.setdefault('engine',engine)
sys.modules['engine.evaluator']=evaluator
from global_local.inference import generate_rois


class Dummy(torch.nn.Module):
    def forward(self,rgb,tir,rgb_semantic,return_features=False):
        device=rgb.device
        pred=[{'bboxes':torch.tensor([[280.,280.,320.,320.]],device=device),
               'conf':torch.tensor([.9],device=device),
               'cls':torch.tensor([0.],device=device)}]
        info={'sampling_scale':torch.ones(1,2,device=device),
              'sampling_translation':torch.zeros(1,2,device=device),
              'coarse_anchor':torch.tensor([[.5,.5]],device=device)}
        return {'pred':pred,'alignment_info':info}


def main():
    batch={'rgb_img':torch.zeros(1,3,768,768),
           'rgb_semantic_img':torch.zeros(1,3,256,256),
           'tir_img':torch.zeros(1,3,256,256),
           'rgb_ratio_pad':[{'ratio':1.0,'pad':(0,0)}],
           'rgb_semantic_ratio_pad':[{'ratio':1/3,'pad':(0,0)}]}
    # predictions in semantic coords for 256-image
    pred=[{'bboxes':torch.tensor([[100.,100.,130.,130.]]),
           'conf':torch.tensor([.8]),'cls':torch.tensor([0.])}]
    scale={'sampling_scale':torch.ones(1,2),'sampling_translation':torch.zeros(1,2),
           'coarse_anchor':torch.tensor([[.4,.5]])}
    rois,_=generate_rois(Dummy(),batch,topk=1,roi_size=128,predictions=pred,alignment_info=scale)
    assert len(rois)==1 and len(rois[0])==1
    p=rois[0][0]
    assert p.rgb.shape==(3,128,128) and p.tir.shape==(3,128,128)
    assert p.from_detector is True
    print('[PASS] global predictions -> two synchronized native 128-pixel ROI tensors')
    empty=[{'bboxes':torch.zeros(0,4),'conf':torch.zeros(0),'cls':torch.zeros(0)}]
    fallback,_=generate_rois(Dummy(),batch,topk=1,roi_size=128,predictions=empty,alignment_info=scale)
    assert len(fallback[0])==1 and not fallback[0][0].from_detector
    print('[PASS] GT-free fallback coarse-anchor ROI')

if __name__=='__main__':main()
