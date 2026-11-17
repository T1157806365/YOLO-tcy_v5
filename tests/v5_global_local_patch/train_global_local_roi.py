#!/usr/bin/env python3
"""Train separate local RGB-T YOLO26m-P2 on predicted-center native HR ROIs.

One GPU: python train_global_local_roi.py --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml
Two GPU: CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc_per_node=2 train_global_local_roi.py --cfg ...
"""
from __future__ import annotations

import argparse,json,math,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from tqdm import tqdm

from engine.runtime import load_config,set_seed,setup_distributed,preprocess_batch,autocast_context,synchronize
from engine.trainer import build_optimizer,build_scheduler
from engine.evaluator import validate_metrics
from global_local.roi_dataset import RGBTLocalROIDataset
from global_local.local_model import build_local_model,LocalEvalAdapter


def loader_for(cfg,split,batch_size,distributed=False,rank=0,world_size=1):
    train=split=='train'
    data=cfg['data'];tcfg=cfg['train']
    ds=RGBTLocalROIDataset(data['roi_root'],split,augment=train,
            fliplr=float(cfg.get('augment',{}).get('fliplr',0.5)))
    sampler=DistributedSampler(ds,num_replicas=world_size,rank=rank,
                               shuffle=train,seed=int(tcfg.get('seed',0))) if distributed else None
    loader=DataLoader(ds,batch_size=batch_size,shuffle=train and sampler is None,
             sampler=sampler,num_workers=int(tcfg.get('workers',2)),pin_memory=True,
             collate_fn=RGBTLocalROIDataset.collate_fn,persistent_workers=int(tcfg.get('workers',2))>0)
    return ds,loader,sampler


def save_model(path,net,optimizer,scheduler,epoch,metric,cfg):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        'architecture':'v5_local_rgbt_add_p2',
        'model_state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
        'config':cfg,'epoch':epoch,'best_val_map5095':float(metric),
        'optimizer_state_dict':optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
    },path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg',required=True)
    parser.add_argument('--smoke',action='store_true',help='1 batch forward/backward only, no checkpoint')
    args=parser.parse_args()
    cfg=load_config(args.cfg);tcfg=cfg['train'];dcfg=cfg['data'];outcfg=cfg['output']
    dev,rank,local_rank,world_size=setup_distributed(tcfg.get('device','5'))
    distributed=world_size>1
    set_seed(int(tcfg.get('seed',0))+rank,bool(tcfg.get('deterministic',True)))
    batch_global=int(tcfg.get('batch',8))
    if batch_global<world_size or batch_global%world_size:
        raise ValueError('train.batch is GLOBAL batch and must be divisible by world_size')
    batch_per_rank=batch_global//world_size
    _,train_loader,sampler=loader_for(cfg,'train',batch_per_rank,distributed,rank,world_size)
    if rank==0:
        _,val_loader,_=loader_for(cfg,'val',int(tcfg.get('val_batch',4)))
    else:val_loader=None
    net=build_local_model(cfg,pretrained=bool(cfg.get('local_model',{}).get('pretrained',True))).to(dev)
    if distributed:
        model=DDP(net,device_ids=[dev.index],output_device=dev.index,
                  find_unused_parameters=bool(tcfg.get('ddp_find_unused_parameters',True)),broadcast_buffers=True)
    else:model=net
    optimizer=build_optimizer(net,cfg)
    epochs=int(tcfg.get('epochs',100));amp=bool(tcfg.get('amp',True))
    scheduler=build_scheduler(optimizer,epochs,lrf=float(tcfg.get('lrf',0.01)))
    scaler=torch.cuda.amp.GradScaler(enabled=(amp and dev.type=='cuda'))
    accumulate=max(int(tcfg.get('accumulate',1)),1)
    grad_clip=float(tcfg.get('grad_clip',10.0))
    project=Path(outcfg.get('project','runs/global_local_roi'))
    out=project/outcfg.get('name','uavcb_local_rgbt_add_100e')
    if rank==0:out.joinpath('weights').mkdir(parents=True,exist_ok=True)
    if distributed:dist.barrier()
    best=-1.0;history=[]
    if rank==0:
        print(f'[LOCAL ROI] input 640x640 RGB+TIR | global frozen/offline | '
              f'world_size={world_size} | batch/rank={batch_per_rank} | accumulate={accumulate}',flush=True)
        print(f'Output: {out.resolve()}',flush=True)
    for ep in range(epochs):
        model.train()
        if sampler is not None:sampler.set_epoch(ep)
        optimizer.zero_grad(set_to_none=True)
        cum=0.0;num_batches=0
        iterable=tqdm(train_loader,desc=f'local train {ep+1}/{epochs}',disable=rank!=0,ncols=110)
        for it,batch in enumerate(iterable):
            batch=preprocess_batch(batch,dev)
            with autocast_context(dev,amp):
                raw,items=model(batch)
                loss=raw.sum()/accumulate
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={ep+1} iter={it}: {loss.item()}')
            scaler.scale(loss).backward()
            if ((it+1)%accumulate==0) or (it+1==len(train_loader)):
                scaler.unscale_(optimizer)
                if grad_clip>0:torch.nn.utils.clip_grad_norm_(net.parameters(),grad_clip)
                scaler.step(optimizer);scaler.update()
                optimizer.zero_grad(set_to_none=True)
            cum+=float(loss.detach().item())*accumulate;num_batches+=1
            if rank==0:iterable.set_postfix({'loss':f'{cum/num_batches:.3f}','GPU':str(dev)})
            if args.smoke:break
        if args.smoke:
            if rank==0:print('[PASS] local model one batch forward/backward/optimizer',flush=True)
            break
        scheduler.step()
        if distributed:dist.barrier()
        if rank==0:
            result=validate_metrics(LocalEvalAdapter(net),val_loader,dev,amp=amp,
                     conf_thres=float(cfg.get('eval',{}).get('local_conf',0.001)),
                     iou_thres=float(cfg.get('eval',{}).get('iou',0.70)))
            metric=float(result['map5095'])
            row={'epoch':ep+1,'train_loss':cum/max(num_batches,1),**result}
            history.append(row)
            print(f"[LOCAL] epoch={ep+1}/{epochs} loss={row['train_loss']:.4f} "
                  f"val_AP50={result['map50']:.4f} val_AP75={result['map75']:.4f} "
                  f"val_AP50:95={metric:.4f} best={max(best,metric):.4f}",flush=True)
            save_model(out/'weights'/'last.pt',net,optimizer,scheduler,ep+1,max(best,metric),cfg)
            if metric>best:
                best=metric
                save_model(out/'weights'/'best.pt',net,optimizer,scheduler,ep+1,best,cfg)
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if distributed:dist.barrier()
    if distributed and dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':main()
