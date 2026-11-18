"""Mock CPU end-to-end train & prediction interface WITHOUT YOLO PT/GPU.

This is a contract test, not a substitute for the real v5 CUDA Smoke Test.
"""
import sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.nn.functional as F

# Stub external v5 imports BEFORE loading package. Test never instantiates real
# YOLO. Local and GMLS fake subnets still exercise actual ROI/model logic.
for name in ('models','models.gtga_bhlr_factory','models.rgbt_model','utils','utils.config','engine','engine.evaluator'):
    x=types.ModuleType(name)
    if '.' not in name:x.__path__=[]
    sys.modules[name]=x
sys.modules['models.gtga_bhlr_factory'].build_model_from_config=lambda *a,**kw:None
sys.modules['models.rgbt_model'].RGBTDetectionModel=type('RGBTDetectionModel',(nn.Module,),{})
sys.modules['utils.config'].resolve_model=lambda *a,**kw:{'weights':'/missing/not/invoked'}

def mocked_postprocess(raw,model,conf_thres,iou_thres):
    return [{'bboxes':raw.new_tensor([[3,3,9,9]]),
             'conf':raw.new_tensor([.85]),'cls':raw.new_tensor([0])}
            for _ in range(raw.shape[0])]
sys.modules['engine.evaluator'].postprocess_predictions=mocked_postprocess

from gmls_roi_single_head.model import GMLSV2ROISingleHead,ROIGuide

class FakeAlignment(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,1)
        self.anchor=nn.Parameter(torch.tensor([[.53,.46]]))
        self.scale=nn.Parameter(torch.tensor([[1.02,.99]]))
        self.trans=nn.Parameter(torch.tensor([[0.,0.]]))
    def align_pyramid(self,rgb_features,tir_features,scale_to_index,
                      rgb_valid_mask,tir_valid_mask,return_debug):
        y=self.conv(rgb_features[0]);b=y.shape[0]
        packet=F.adaptive_avg_pool2d(torch.sigmoid(y), (8,8))
        return [],{
            'multi_target_guidance':packet,
            'coarse_anchor':self.anchor.expand(b,-1).clamp(0,1),
            'sampling_scale':self.scale.expand(b,-1),
            'sampling_translation':self.trans.expand(b,-1),
        }

class FakeGlobal(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,3,padding=1)
        self.alignment=FakeAlignment()
        self.scale_to_index={3:0}
        self.alignment_loss_enabled=True
        self.lambda_alignment=.2
    def forward_rgb_backbone(self,x):return [self.conv(x)]
    def forward_tir_backbone(self,x):return [self.conv(x)]
    def build_alignment_valid_masks_from_batch(self,batch):return None,None
    def _alignment_auxiliary_loss(self,batch,info):
        st=info['multi_target_guidance'].mean() + info['coarse_anchor'].mean()*.01
        return {'total':st,'anchor':st,'coarse':st,'scale':st,'targetness':st,
                'v2_quality':st,'v2_route':st,'v2_gmls_gt_quality':st,'v2_dmls_gt_quality':st}

class FakeLocal(nn.Module):
    def __init__(self):super().__init__();self.conv=nn.Conv2d(6,1,3,padding=1)
    def forward(self,b):
        x=self.conv(torch.cat((b['rgb_img'],b['tir_img']),1))
        return x.square().mean(),x.new_tensor([.2,.3,.1])
    def predict(self,rgb,tir):return self.conv(torch.cat((rgb,tir),1))


def run():
    torch.manual_seed(0)
    net=GMLSV2ROISingleHead.__new__(GMLSV2ROISingleHead)
    nn.Module.__init__(net)
    net.global_encoder=FakeGlobal()
    net.guide=ROIGuide(hidden=8,temperature=.12)
    net.detector=FakeLocal()
    net.cfg={'train':{'gt_assist_start':.0,'gt_assist_epochs':1},
             'loss':{'roi':{'sigma_px':2.5}}}
    net.anchor_mix=.3;net.roi_size=20;net.lambda_guide=.12
    net.lambda_center=1.;net.lambda_local=1.
    net.center_grad_scale=.2;net.geometry_grad_scale=.2
    net.min_visible=.5;net.epoch=0
    meta_h=[{'ratio':3.,'pad':(0,0)}];meta_s=[{'ratio':1.,'pad':(0,0)}]
    batch={
        'rgb_img':torch.rand(1,3,60,60),
        'rgb_semantic_img':torch.rand(1,3,20,20),
        'tir_img':torch.rand(1,3,20,20),
        'rgb_ratio_pad':meta_h,'rgb_semantic_ratio_pad':meta_s,
        'rgb_cls':torch.zeros(1,1),
        'rgb_bboxes':torch.tensor([[.5,.5,.13,.13]]),
        'rgb_semantic_bboxes':torch.tensor([[.5,.5,.13,.13]]),
        'rgb_batch_idx':torch.zeros(1,dtype=torch.long),
    }
    net.train()
    loss,stats=net(batch)
    assert torch.isfinite(loss) and 'roi_positive' in stats
    loss.backward()
    assert net.guide.conv[-1].weight.grad is not None
    assert net.global_encoder.alignment.scale.grad is not None
    assert net.detector.conv.weight.grad is not None
    net.eval()
    # Deliberately OMIT labels: inference must not depend on GT.
    inference={k:v for k,v in batch.items() if k not in (
        'rgb_cls','rgb_bboxes','rgb_semantic_bboxes','rgb_batch_idx')}
    preds,details=net.predict(inference)
    assert len(preds)==1 and preds[0]['bboxes'].shape==(1,4)
    print('[PASS] one-head ROI mock joint train gradients + prediction without any GT inputs')

if __name__=='__main__':run()
