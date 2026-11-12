"""Confirm initialized new gate reproduces same stable baseline P3 tensor, on actual UAV-CB image."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import yaml
from engine.runtime import select_device,preprocess_batch
from tri_fusion.train import make_loader,load_base_with_gate

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--device',default='3')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.cfg).read_text(encoding='utf-8'))
    device=select_device(args.device)
    model=load_base_with_gate(cfg,device).eval()
    ds,loader=make_loader(cfg,'val',1,0)
    data=preprocess_batch(next(iter(loader)),device)
    core=model.core
    f=core.forward_features(data['rgb_img'],data['tir_img'],
                            rgb_semantic=data['rgb_semantic_img'])
    p3=int(core.scale_to_index[3])
    new=f['fused_backbone'][p3].detach().float().clone()
    old_enabled=core.tri_gate_enabled
    core.tri_gate_enabled=False
    try:
        fo=core.forward_features(data['rgb_img'],data['tir_img'],
                                rgb_semantic=data['rgb_semantic_img'])
    finally:core.tri_gate_enabled=old_enabled
    old=fo['fused_backbone'][p3].float()
    diff=(new-old).abs().max().item()
    print('[PREFLIGHT] max |P3_new_initial - P3_original| =',diff)
    if diff>1e-5:
        raise AssertionError(f'Non-identity at initialization; diff={diff}')
    if any(abs(float(v)-1.0)>1e-6 for k,v in core.tri_gate.last_stats.items() if k.startswith('w_')):
        raise AssertionError('Zero-init gate weights are not 1')
    print('[PASS] Original Post-Fusion P3 and new gate at initialization agree')

if __name__=='__main__':main()
