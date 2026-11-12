"""Train ONLY the new P3 attention for 20e on top of the full 61.4162% RGB-T checkpoint.

Does not change original GTGA, BHLR, RGB/TIR backbones, mono heads or neck weights.
No use of ground-truth or teacher weights at inference; no new detection head.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import yaml
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from engine.runtime import select_device, preprocess_batch, autocast_context
from engine.evaluator import validate_metrics
from ga_monoalign_pcgrad.factory import build


def make_loader(cfg, split, batch_size, workers):
    d=cfg['data']
    ds=build_dual_resolution_dataset(
        rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=split,
        rgb_high_imgsz=int(d['rgb_imgsz']),
        rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=(split=='train'),tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(
        ds,batch_size=batch_size,workers=workers,shuffle=(split=='train'),
        pin_memory=True)
    return ds,loader


def load_base_with_gate(cfg,device):
    settings=cfg['tri_experiment']
    ck=torch.load(settings['base_weights'],map_location='cpu',weights_only=False)
    original=ck['config']['model']
    if original.get('bhlr',{}).get('injection_position')!='post_fusion':
        raise ValueError('Not a Post-Fusion BHLR checkpoint')
    sd=ck['model_state_dict']
    if any('aghr' in name.lower() or 'accf' in name.lower() or 'tri_gate' in name.lower()
           for name in sd):
        raise ValueError('Base checkpoint must be original 61.4162%, not new experiment')
    model=build(cfg,device='cpu',load_pretrained=False)
    expected={key for key in model.state_dict() if key.startswith('core.tri_gate.')}
    if not expected:
        raise RuntimeError('Tri-gate was not constructed; verify installer / YAML')
    report=model.load_state_dict(sd,strict=False)
    missing=set(report.missing_keys)
    if missing!=expected or report.unexpected_keys:
        raise RuntimeError('Original checkpoint incompatible with patched model! '
                           f'missing={sorted(missing)}, expected={sorted(expected)}, '
                           f'unexpected={report.unexpected_keys}')
    # Freeze ALL previously trained model weights, including GTGA/BHLR/Mono.
    for param in model.parameters():param.requires_grad_(False)
    for param in model.core.tri_gate.parameters():param.requires_grad_(True)
    model.to(device)
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('No trainable tri-gate parameters')
    print('[BASE LOADED] strict old-state keys checked; new module only missing; '
          'GTGA/BHLR/YOLO frozen; source=',settings['base_weights'],flush=True)
    return model


def set_training_state(model):
    model.train()  # YOLO criterion needs train-form output from Detect.
    for layer in model.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm):
            layer.eval()  # preserve frozen running statistics exactly
    model.core.tri_gate.train()


def train_one(cfg):
    settings=cfg['tri_experiment']
    if settings['variant']=='A':
        raise ValueError('A baseline is evaluation only; do not train it')
    if cfg['model'].get('tri_gate',{}).get('enabled') is not True:
        raise ValueError('Must enable tri_gate to train new module')
    device=select_device(str(settings['device']))
    seed=int(settings.get('seed',0))
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    output=Path(cfg['output']['project'])/cfg['output']['name']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite: {output}')
    output.mkdir(parents=True,exist_ok=True)
    (output/'weights').mkdir(exist_ok=True)
    (output/'source_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
    _,train_loader=make_loader(cfg,'train',int(settings['batch']),int(settings.get('workers',4)))
    _,val_loader=make_loader(cfg,'val',max(1,min(4,int(settings['batch']))),int(settings.get('workers',4)))
    model=load_base_with_gate(cfg,device)
    core=model.core
    core.criterion=core.init_criterion()
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=float(settings['lr']),weight_decay=float(settings.get('weight_decay',0.0001)))
    amp=bool(settings.get('amp',True)) and device.type=='cuda'
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    accumulate=max(1,int(settings.get('accumulate',8)))
    clip=float(settings.get('grad_clip',5.0))
    epoch_count=int(settings['epochs'])
    best=-1.0
    for epoch in range(1,epoch_count+1):
        set_training_state(model)
        running=0.0;steps=0;skips=0
        opt.zero_grad(set_to_none=True)
        progress=tqdm(enumerate(train_loader),total=len(train_loader),desc=f'[TRI] {epoch}/{epoch_count}')
        for i,batch in progress:
            batch=preprocess_batch(batch,device)
            with autocast_context(device,amp):
                raw=core.predict(batch['rgb_img'],batch['tir_img'],
                                 rgb_semantic=batch['rgb_semantic_img'])
                yolo_batch={'img':batch['rgb_semantic_img'],
                            'cls':batch['rgb_cls'],
                            'bboxes':batch['rgb_semantic_bboxes'],
                            'batch_idx':batch['rgb_batch_idx']}
                vec,_=core.criterion(raw,yolo_batch)
                bs=max(1,int(batch['rgb_img'].shape[0]))
                det_loss=vec.sum()/bs
                # Mild identity prior; never forces either modality to be used.
                reg=float(settings.get('identity_reg',0.0))*(
                    core.tri_gate.logit.weight.float().square().mean()+
                    core.tri_gate.logit.bias.float().square().mean())
                loss=(det_loss+reg)/accumulate
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}, batch {i}')
            scaler.scale(loss).backward()
            running+=float(det_loss.detach())
            sync=(i+1)%accumulate==0 or i+1==len(train_loader)
            if sync:
                scaler.unscale_(opt)
                norms=torch.nn.utils.clip_grad_norm_(trainable,clip,error_if_nonfinite=False)
                if not torch.isfinite(norms):
                    raise FloatingPointError(f'Nonfinite attention gradient at epoch {epoch}, batch {i}')
                before=float(scaler.get_scale())
                scaler.step(opt);scaler.update()
                after=float(scaler.get_scale())
                if after<before:skips+=1
                else:steps+=1
                opt.zero_grad(set_to_none=True)
            if i%10==0:progress.set_postfix(loss=f'{float(det_loss):.4f}',updates=steps,skips=skips)
        if steps<1:
            raise RuntimeError('Entire epoch skipped all optimizer steps; check AMP / gradients')
        model.eval()
        metric=validate_metrics(model,val_loader,device,amp=amp,
                                conf_thres=0.001,iou_thres=0.70,max_det=300,preview_num=0)
        ap=float(metric['map5095'])
        stats=dict(core.tri_gate.last_stats)
        print(f'[TRI MONITOR] epoch={epoch} AP5095={ap:.6f} '
              f'AP75={metric["map75"]:.6f} weights={stats} '
              f'updates={steps} skipped={skips} det={running/max(len(train_loader),1):.6f}',flush=True)
        history_path=output/'history.jsonl'
        with history_path.open('a',encoding='utf-8') as fh:
            fh.write(json.dumps({'epoch':epoch,**metric,**stats,'updates':steps,'skipped':skips},ensure_ascii=False)+'\n')
        save={'model_state_dict':model.state_dict(),'config':cfg,
              'epoch':epoch,'stage':'tri_gate_module_only','best_map5095':max(best,ap),
              'model_name':model.model_name,'fusion':model.fusion_type,
              'nc':model.nc,'names':model.names}
        torch.save(save,output/'weights'/'last.pt')
        if ap>best:
            best=ap
            torch.save(save,output/'weights'/'best.pt')
    print('[DONE] best validation AP50:95',best)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.cfg).read_text(encoding='utf-8'))
    if args.check:
        settings=cfg['tri_experiment']
        if settings['variant']=='A':print('[CHECK] A baseline: validation only');return
        if not Path(settings['base_weights']).is_file():raise FileNotFoundError(settings['base_weights'])
        print('[CHECK] config valid, baseline exists, no output overwrite intended')
        return
    train_one(cfg)

if __name__=='__main__':main()
