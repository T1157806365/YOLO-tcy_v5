"""Local auxiliary supervision, without a new head or gradients into crop coordinates."""
from contextlib import contextmanager
import torch
from torch import nn


def subset_labels(labels,ids):
    n=len(labels['rgb_img']);bi=labels['rgb_batch_idx'].long().flatten()
    remap=torch.full((n,),-1,device=bi.device,dtype=torch.long)
    remap[ids]=torch.arange(len(ids),device=bi.device)
    mapped=remap[bi];keep=mapped>=0
    return {'rgb_img':labels['rgb_img'].index_select(0,ids).detach(),
            'tir_img':labels['tir_img'].index_select(0,ids).detach(),
            'rgb_cls':labels['rgb_cls'][keep].detach(),
            'rgb_bboxes':labels['rgb_bboxes'][keep].detach(),
            'rgb_batch_idx':mapped[keep].to(labels['rgb_batch_idx'].dtype)}


@contextmanager
def freeze_bn_statistics(module):
    # Keep Detect in train mode for loss, only disable running-stat updates.
    bn=[(m,m.training) for m in module.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    try:
        for m,_ in bn:m.eval()
        yield
    finally:
        for m,flag in bn:m.train(flag)
