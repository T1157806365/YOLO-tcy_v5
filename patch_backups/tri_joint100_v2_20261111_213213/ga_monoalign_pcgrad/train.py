"""300+200 epochs original GMLS-v2 scratch training, two mono heads, GT+teacher KD, PCGrad."""
from __future__ import annotations
import argparse,json,math,os,shutil,subprocess,sys
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.runtime import (load_config,parse_device_ids,setup_distributed,
    cleanup_distributed,distributed_barrier,is_main_process,set_seed,
    preprocess_batch,autocast_context,make_grad_scaler,distributed_mean,
    distributed_mean_dict,broadcast_metrics)
from engine.evaluator import validate_metrics
from engine import trainer as tr
from .factory import build,validate_cfg
from .metrics import mono_metrics,geometry_iou
from .rich_logging import stage_banner, progress_description, epoch_report, write_history_csv


def as_float_dict(items):
    return tr.normalize_loss_items(items)


def broadcast_all_ga_metrics(metric, device):
    """Broadcast ALL validation metrics, not just the 7 standard YOLO metrics.

    engine.runtime.broadcast_metrics discards align_iou and mono auxiliary
    metrics. GPU rank 0 owns the full result; broadcast_object_list makes
    the same dictionary available to all ranks. One call per epoch only.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload = [metric if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(payload, src=0, device=device)
        metric = payload[0]
    if not isinstance(metric, dict):
        raise TypeError(f"Expected validation metrics dict, got {type(metric)}")
    required = ('map5095', 'align_iou', 'align_pairs', 'align_zero')
    missing = [k for k in required if k not in metric]
    if missing:
        raise RuntimeError(
            f"Validation metrics missing {missing}; available={list(metric)}"
        )
    if not math.isfinite(float(metric['align_iou'])) or int(metric['align_pairs']) < 1:
        raise RuntimeError(
            f"Invalid alignment validation statistics: "
            f"align_iou={metric['align_iou']}, align_pairs={metric['align_pairs']}"
        )
    return metric


def save(path,net,cfg,stage,epoch,global_epoch,optimizer,scheduler,scaler,best_ap,best_joint):
    if not is_main_process():return
    model=net.module if isinstance(net,DDP) else net
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'epoch':epoch,'stage':stage,'global_epoch':global_epoch,
        'model_state_dict':model.state_dict(),'config':cfg,'optimizer_state_dict':optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),
        'best_map5095':best_ap,'best_joint_score':best_joint,
        'model_name':model.model_name,'fusion':model.fusion_type,
        'nc':model.nc,'names':model.names},path)


def run_stage(net,cfg,train_loader,val_loader,device,out,stage,nepoch,lr0,
              patience,offset,global_best):
    raw=net.module if isinstance(net,DDP) else net
    optimizer=tr.build_optimizer(net,cfg,lr0=lr0)
    lrf=float(cfg['train'].get('lrf',.01) if stage=='stage1' else cfg['train']['stage2'].get('lrf',.01))
    # multi_step_100_lr: preserve cosine for legacy YAMLs.
    train_cfg = cfg['train']
    lr_schedule = str(train_cfg.get('scheduler', 'cosine')).strip().lower()
    if stage == 'stage1' and lr_schedule == 'multistep':
        milestones = [int(x) for x in train_cfg.get('lr_milestones', [50, 80])]
        gamma_step = float(train_cfg.get('lr_gamma', 0.2))
        if not (0 < gamma_step < 1):
            raise ValueError('Invalid train.lr_gamma')
        if any(x < 1 or x >= nepoch for x in milestones):
            raise ValueError('Invalid train.lr_milestones')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sorted(milestones), gamma=gamma_step
        )
    else:
        scheduler=tr.build_scheduler(optimizer,nepoch,lrf)
    scaler=make_grad_scaler(device,bool(cfg['train'].get('amp',True)))
    amp=bool(cfg['train'].get('amp',True))
    accumulation=int(cfg['train'].get('accumulate',1))
    clip=float(cfg['train'].get('grad_clip',10.))
    gamma=float(cfg['monoalign'].get('save_joint_gamma',.10))
    count_bad=0;best=-1.;joint_best=-math.inf;best_path=out/'weights'/f'{stage}_best.pt'
    history=[];mono_interval=int(cfg['monoalign'].get('mono_val_every',5))
    latest_mono=None
    if is_main_process():
        print(stage_banner(stage,nepoch,lr0,lrf,patience,cfg),flush=True)
    for epoch in range(1,nepoch+1):
        raw.set_progress(stage,epoch)
        # Load frozen teachers ONLY when KD activates (after Epoch 60).
        # Avoid retaining two full teachers in VRAM during baseline warmup.
        if raw.schedule()[1]>0 and raw._teachers is None:
            raw.attach_teachers(device)
        net.train()
        # Detectors + auxiliary heads train; teachers are outside nn.Module.
        sampler=getattr(train_loader,'sampler',None)
        if hasattr(sampler,'set_epoch'):sampler.set_epoch(epoch+offset)
        optimizer.zero_grad(set_to_none=True)
        sumloss=0.;counts=0;logitems={}
        from tqdm import tqdm
        it=tqdm(train_loader,disable=not is_main_process(),desc=f'{stage} {epoch}/{nepoch}')
        for bi,batch in enumerate(it):
            batch=preprocess_batch(batch,device)

            # PCGrad correction must match the scaled backward gradient.
            # Actual backward:
            # scaler.scale(loss_vec.sum() / (batch_size * accumulation))
            raw._pcgrad_backward_scale = (
                float(scaler.get_scale())
                / (
                    max(1, int(batch['rgb_img'].shape[0]))
                    * accumulation
                )
            )

            # no_sync only on intermediate accumulation steps.
            import contextlib
            sync_now=(bi+1)%accumulation==0 or (bi+1)==len(train_loader)
            ctx=net.no_sync() if isinstance(net,DDP) and not sync_now else contextlib.nullcontext()
            with ctx:
                with autocast_context(device,amp):
                    loss_vec,log=net(batch)
                    # Per-image scaling eliminates hidden per-rank batch factors;
                    # divide by accumulation to preserve effective global batch.
                    loss=loss_vec.sum()/(max(1,batch['rgb_img'].shape[0])*accumulation)
                scaler.scale(loss).backward()
            sumloss+=float(loss_vec.sum().detach())/max(1,batch['rgb_img'].shape[0])
            counts+=1
            logitems=tr.update_running_losses(logitems,as_float_dict(log),bi)
            if is_main_process() and (bi % 4 == 0 or sync_now):
                it.set_description_str(progress_description(
                    stage,epoch,nepoch,device,logitems,
                    int(batch["rgb_cls"].shape[0]),
                    cfg["data"].get("rgb_imgsz",1920),
                    cfg["data"].get("tir_imgsz",640)),refresh=False)
            if sync_now:
                scaler.unscale_(optimizer)
                if clip>0:torch.nn.utils.clip_grad_norm_(net.parameters(),clip)
                scaler.step(optimizer);scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        trainloss=distributed_mean(sumloss/max(1,counts),device)
        logitems=distributed_mean_dict(logitems,device)
        distributed_barrier()
        metric=None
        if is_main_process():
            raw.eval()
            metric=validate_metrics(raw,val_loader,device,amp=amp,
                conf_thres=.001,iou_thres=.7,max_det=300,preview_num=0)
            geom=geometry_iou(raw,val_loader,device,amp=False,
                max_images=int(cfg['monoalign'].get('alignment_val_images',96)))
            metric.update(geom)
            if mono_interval>0 and (epoch%mono_interval==0 or epoch==1):
                metric.update(mono_metrics(raw,val_loader,device,amp))
        metric=broadcast_all_ga_metrics(metric,device)
        score=float(metric['map5095'])
        giou=float(metric['align_iou'])
        joint=score+gamma*giou if math.isfinite(giou) else -math.inf
        is_better=score>best+float(cfg['train'].get('early_stop_min_delta',1e-6))
        if is_better:best=score;count_bad=0
        else:count_bad+=1
        global_best=max(global_best,score)
        if is_main_process():
            last=out/'weights'/f'{stage}_last.pt'
            save(last,net,cfg,stage,epoch,offset+epoch,optimizer,scheduler,scaler,best,joint_best)
            shutil.copy2(last,out/'weights'/'last.pt')
            if is_better:
                shutil.copy2(last,best_path)
                if score>=float(global_best)-1e-12:
                    shutil.copy2(last,out/'weights'/'best.pt')
            if joint>joint_best:
                joint_best=joint
                shutil.copy2(last,out/'weights'/f'{stage}_joint_best.pt')
                if joint>=float(getattr(run_stage,'_global_joint_best',-math.inf)):
                    setattr(run_stage,'_global_joint_best',joint)
                    shutil.copy2(last,out/'weights'/'joint_best.pt')
            row={'stage':stage,'epoch':epoch,'global_epoch':offset+epoch,
                'train_loss_per_image':trainloss,'lr':optimizer.param_groups[0]['lr'],
                **logitems,**metric}
            history.append(row)
            (out/f'history_{stage}.json').write_text(json.dumps(history,indent=2,ensure_ascii=False),encoding='utf-8')
            write_history_csv(out/f"history_{stage}.csv",history)
            if 'rgb_only_map5095' in metric or 'tir_only_map5095' in metric:
                latest_mono=dict(metric)
                latest_mono['_epoch']=epoch
            print(epoch_report(stage=stage,epoch=epoch,nepoch=nepoch,
                trainloss=trainloss,lr=optimizer.param_groups[0]['lr'],
                logitems=logitems,metric=metric,
                best=best,global_best=global_best,joint_best=joint_best,
                no_improve=count_bad,patience=patience,
                mono_interval=mono_interval,latest_mono=latest_mono),flush=True)
        distributed_barrier()
        # Do not let ordinary patience stop training BEFORE KD begins.
        minimum=int(cfg['monoalign'].get('minimum_stage1_epochs',130)) if stage=='stage1' else 0
        if count_bad>=patience and epoch>=minimum:break
    return {'best':best,'global_best':global_best,'epochs_ran':epoch,'best_path':best_path}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True);p.add_argument('--check',action='store_true')
    args=p.parse_args()
    cfg=validate_cfg(load_config(args.cfg),need_teachers=True)
    output=Path(cfg['output']['project'])/cfg['output']['name']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output nonempty, refusing overwrite: {output}')
    devices=parse_device_ids(cfg['train']['device'])
    if args.check:
        print('[CHECK] Student scratch; GMLS-v2 unchanged; teachers frozen; output',output)
        return
    if len(devices)>1 and int(os.environ.get('WORLD_SIZE','1'))==1:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(map(str,devices))
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
             f'--nproc_per_node={len(devices)}','-m','ga_monoalign_pcgrad.train',
             '--cfg',str(Path(args.cfg).resolve())]
        print('[DDP] Physical GPUs:',devices,flush=True)
        subprocess.run(cmd,check=True,env=env)
        return
    device,rank,local_rank,world=setup_distributed(cfg['train']['device'])
    try:
        trcfg=cfg['train'];set_seed(int(trcfg.get('seed',0)),bool(trcfg.get('deterministic',True)))
        loc,glob,_=tr._resolve_batch_size(cfg,world)
        _,trainloader=tr._build_data(cfg,'train',True,True,
            batch_size=loc,distributed=world>1,rank=rank,world_size=world)
        _,valloader=tr._build_data(cfg,'val',False,False,
            batch_size=int(trcfg.get('val_batch',2)),distributed=False,rank=0,world_size=1) if is_main_process() else (None,None)
        student=build(cfg,device=device)
        # Teachers are not nn.Module children and are never optimized.
        # Teachers are loaded lazily when the stage schedule turns on KD.
        if is_main_process():
            (output/'weights').mkdir(parents=True,exist_ok=False)
        distributed_barrier()
        net=DDP(student,device_ids=[int(device.index)],output_device=int(device.index),
            find_unused_parameters=True,broadcast_buffers=True) if world>1 else student
        print(f'[TRAIN] rank={rank}, device={device}, batch/rank={loc}, '
              f'global batch={glob}, accumulate={trcfg.get("accumulate",1)}, '
              f'GMLS-v2 original FULL, official YOLO pretrained={cfg["model"]["pretrained"]}',flush=True)
        one=run_stage(net,cfg,trainloader,valloader,device,output,'stage1',int(trcfg['epochs']),
                      float(trcfg['lr0']),int(trcfg.get('patience',50)),0,-1.)
        distributed_barrier()
        ck=torch.load(one['best_path'],map_location='cpu',weights_only=False)
        student.load_state_dict(ck['model_state_dict'],strict=True)
        two=None
        if trcfg.get('stage2',{}).get('enabled',True):
            st=trcfg['stage2']
            two=run_stage(net,cfg,trainloader,valloader,device,output,'stage2',int(st['epochs']),
                float(st['lr0']),int(st.get('patience',50)),one['epochs_ran'],one['global_best'])
        if is_main_process():
            summary={'stage1_best':one['best'],'stage2_best':two['best'] if two else None,
                'global_best':two['global_best'] if two else one['global_best'],
                'student_scratch':not cfg['model']['pretrained'],
                'official_yolo_pretrained':bool(cfg['model']['pretrained']),
                'original_gmlsv2_preserved':True}
            (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print('[TRAIN COMPLETE]',summary,flush=True)
        distributed_barrier()
    finally:
        cleanup_distributed()

if __name__=='__main__':main()
