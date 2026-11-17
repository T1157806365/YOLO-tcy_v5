"""Mock full GMLS-v2+ROI joint loss/forward on CPU without Ultralytics.

It validates only plumbing, NOT real YOLO26 or CUDA/DDP behavior.
"""
import sys
from pathlib import Path
import types
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

factory = types.ModuleType('models.gtga_bhlr_factory')
base_model = types.ModuleType('models.rgbt_model')
eval_module = types.ModuleType('engine.evaluator')
engine_pkg = types.ModuleType('engine')
engine_pkg.__path__ = []

def mock_prediction(n, device):
    return [{'bboxes':torch.tensor([[60.,60.,70.,70.]],device=device),
             'conf':torch.tensor([0.75],device=device),
             'cls':torch.tensor([0.],device=device)} for _ in range(n)]

class Global(nn.Module):
    def __init__(self):
        super().__init__()
        self.w=nn.Parameter(torch.tensor(1.))
        self.criterion=None
        self.alignment_loss_enabled=True
        self.lambda_alignment=0.2
        self.args=types.SimpleNamespace()
        self.end2end=True
    def init_criterion(self):
        return lambda pred,batch: (self.w.square().reshape(1), {})
    def build_alignment_valid_masks_from_batch(self, batch):
        return None,None
    def predict(self, rgb, tir, rgb_semantic, **kwargs):
        n=len(rgb);dev=rgb.device
        return {'pred':mock_prediction(n,dev),
            'alignment_info':{'coarse_anchor':torch.full((n,2),.5,device=dev),
                'sampling_scale':torch.ones(n,2,device=dev),
                'sampling_translation':torch.zeros(n,2,device=dev)}}
    def _alignment_auxiliary_loss(self, batch, info):
        return {'total':self.w.square()*.01}

class Local(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.w=nn.Parameter(torch.tensor(1.))
        self.args=types.SimpleNamespace()
        self.criterion=None
        self.end2end=True
    def forward(self,batch):
        return self.w.square().reshape(1), {}
    def predict(self,rgb,tir):
        return mock_prediction(len(rgb),rgb.device)

factory.build_model_from_config=lambda cfg, pretrained=False,verbose=False: Global()
base_model.RGBTDetectionModel=Local
eval_module.postprocess_predictions=lambda pred, model, conf_thres=.001, iou_thres=.7:pred
sys.modules['models']=types.ModuleType('models')
sys.modules['models'].__path__=[]
sys.modules['models.gtga_bhlr_factory']=factory
sys.modules['models.rgbt_model']=base_model
sys.modules['engine']=engine_pkg
sys.modules['engine.evaluator']=eval_module

from global_local_scratch.joint_model import GMLSV2GlobalLocalScratch

def make_cfg():
    return {'model':{'name':'yolo26m','pretrained':False,
                    'global_local':{'enabled':True,'roi_size':96,
                        'topk':2,'local_pretrained':False}},
            'train':{'resume':None},'loss':{'box':7.5,'cls':.5,'dfl':1.5}}

def main():
    bsz=1;rgb=torch.zeros(bsz,3,384,384);sem=torch.zeros(bsz,3,128,128)
    tir=torch.zeros(bsz,3,128,128)
    batch={
       'rgb_img':rgb,'rgb_semantic_img':sem,'tir_img':tir,
       'rgb_cls':torch.zeros(1,1),
       'rgb_bboxes':torch.tensor([[.5,.5,.1,.1]]),
       'rgb_semantic_bboxes':torch.tensor([[.5,.5,.1,.1]]),
       'rgb_batch_idx':torch.tensor([0]),
       'tir_bboxes':torch.tensor([[.5,.5,.1,.1]]),
       'tir_batch_idx':torch.tensor([0]),
       'rgb_ratio_pad':[{'ratio':1.,'pad':(0,0)}],
       'rgb_semantic_ratio_pad':[{'ratio':1/3,'pad':(0,0)}],
    }
    net=GMLSV2GlobalLocalScratch(make_cfg())
    loss,parts=net(batch)
    assert float(loss)>0, loss
    loss.backward()
    assert net.global_model.w.grad is not None
    assert net.local_model.w.grad is not None
    net.eval()
    inf=net.predict_global_local(batch,conf=.001)
    assert len(inf['global'])==len(inf['fused'])==len(inf['local'])==1
    assert inf['roi_count']>=1
    print('[PASS] full mocked from-scratch joint backward and GT-free inference')

if __name__=='__main__':main()
