"""Reuses the existing v5 dual-backbone ADD architecture (no BHLR/GTGA)."""
from __future__ import annotations
from pathlib import Path
import torch
from models.rgbt_model import RGBTDetectionModel


def build_local_model(cfg: dict, *, pretrained: bool) -> RGBTDetectionModel:
    spec=cfg.get('local_model',{})
    if str(spec.get('fusion','add')).lower()!='add':
        raise ValueError('local RGB-T version is restricted to original ADD baseline')
    m=RGBTDetectionModel(
        model_name=str(spec.get('name','yolo26m-p2')),
        nc=1, pretrained=bool(pretrained), fusion='add',
        fusion_indices=[int(x) for x in spec.get('fusion_indices',[4,6,10])],
        align_mode='bilinear', names={0:'UAV'},
        p2_head_pretrain=bool(spec.get('p2_head_pretrain',False) and pretrained),
        verbose=bool(spec.get('verbose',False)),
    )
    lc=cfg.get('loss',{})
    m.args.box=float(lc.get('box',7.5))
    m.args.cls=float(lc.get('cls',0.5))
    m.args.dfl=float(lc.get('dfl',1.5))
    m.criterion=None
    return m


def load_local_checkpoint(path: str | Path, device: torch.device):
    path=Path(path).expanduser().resolve()
    obj=torch.load(path,map_location='cpu',weights_only=False)
    cfg=obj['config']
    model=build_local_model(cfg,pretrained=False)
    raw=obj.get('model_state_dict',obj.get('local_model_state_dict'))
    if raw is None:
        raise KeyError('Checkpoint missing model_state_dict')
    if raw and all(k.startswith('module.') for k in raw):
        raw={k[7:]:v for k,v in raw.items()}
    model.load_state_dict(raw,strict=True)
    model.to(device).eval()
    return model,obj,cfg


class LocalEvalAdapter(torch.nn.Module):
    """v5 engine.evaluator passes rgb_semantic kwarg, the base model does not."""
    def __init__(self, model):
        super().__init__()
        self.inner=model
        self.names=model.names
        self.nc=model.nc
        self.end2end=model.end2end

    def forward(self,rgb,tir,rgb_semantic=None):
        return self.inner(rgb,tir)
