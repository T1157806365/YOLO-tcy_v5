"""Dual-GPU joint optimization and exact global val statistics for one-head GMLS ROI.

Evaluation is split across all ranks with non-overlapping examples; raw image
statistics are gathered once and AP is computed on rank0 (NOT averaged AP).
"""
from __future__ import annotations
import contextlib, csv, json, os, time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
from ultralytics.utils.metrics import DetMetrics
from engine.runtime import load_config, set_seed, setup_distributed, preprocess_batch, autocast_context
from engine.trainer import build_optimizer, build_scheduler
from engine.evaluator import prepare_semantic_ground_truth, process_single_image
from datasets.rsdt_dataset import build_rsdt_dataset
from .geometry import roi_gt_coverage
from .model import GMLSV2ROISingleHead


def build_loader(cfg,split,batch,workers,rank=0,world=1):
    dc=cfg['data'];aug=cfg.get('augment',{});tc=cfg['train']
    dataset=build_rsdt_dataset(
        rgb_yaml=dc['rgb'],tir_yaml=dc['tir'],split=split,
        rgb_high_imgsz=int(dc.get('rgb_imgsz',1920)),
        rgb_semantic_imgsz=int(dc.get('rgb_semantic_imgsz',640)),
        tir_imgsz=int(dc.get('tir_imgsz',640)),
        pair_mode=str(dc.get('pair_mode','relative')),
        augment=(split=='train'),fliplr=float(aug.get('fliplr',.5)),
        flipud=float(aug.get('flipud',0)),tir_channels=3,strict_pair=True)
    sampler=None
    if world>1 and split=='train':
        sampler=DistributedSampler(dataset,world,rank,shuffle=True,
                                   drop_last=False,seed=int(tc.get('seed',0)))
    if world>1 and split!='train':
        # Split with NO sampler padding/duplicates: exact AP and ROI coverage.
        dataset=Subset(dataset,list(range(rank,len(dataset),world)))
    loader=DataLoader(dataset,batch_size=int(batch),
                      shuffle=(split=='train' and sampler is None),
                      sampler=sampler,num_workers=int(workers),
                      collate_fn=(dataset.dataset.collate_fn if isinstance(dataset,Subset) else dataset.collate_fn),
                      pin_memory=True,persistent_workers=int(workers)>0,
                      drop_last=False)
    return loader,sampler


def _asfloat(x):
    if torch.is_tensor(x): return float(x.detach().float().mean().item())
    return float(x)


def _metric_row(metric:DetMetrics):
    metric.process(plot=False)
    pr,re,ap50,ap=metric.mean_results()
    return {'P':float(pr),'R':float(re),'AP50':float(ap50),
            'AP75':float(metric.box.map75),'AP50:95':float(ap)}


@torch.inference_mode()
def evaluate(net,loader,device,cfg,*,rank=0,world=1,smoke=False,preview_dir=None):
    net.eval();ecfg=cfg.get('eval',{}); conf=float(ecfg.get('conf',.001));iou=float(ecfg.get('iou',.7))
    amp=bool(cfg['train'].get('amp',True)) and device.type=='cuda'
    iouv=torch.linspace(.5,.95,10,device=device)
    records=[];hit=total=0;valid=0.;roi_count=0
    align_iou_sum=align_center_sum=align_num=0.
    guide_err_sum=0.;guide_err_num=0
    val_t0=time.perf_counter()
    preview_count=0
    pbar=tqdm(loader,desc='Validation',disable=(rank!=0),leave=False,ncols=125)
    for batch in pbar:
        batch=preprocess_batch(batch,device)
        with autocast_context(device,amp):
            preds,ex=net.predict(batch,conf=conf,iou=iou)
        h,n=roi_gt_coverage(batch,ex['roi'],min_visible=float(cfg['model']['roi_single_head'].get('min_gt_visible',.6)))
        hit+=h;total+=n
        b=len(preds)
        roi_count+=b;valid+=float(ex['roi'].tir_valid.sum().item())
        sem_hw=batch['rgb_semantic_img'].shape[-2:]
        for k,pred in enumerate(preds):
            gt=prepare_semantic_ground_truth(batch,k)
            target=gt['cls'].detach().float().cpu().numpy()
            tp=process_single_image(pred,gt,iouv)
            records.append({'tp':tp,'conf':pred['conf'].detach().float().cpu().numpy(),
                            'pred_cls':pred['cls'].detach().float().cpu().numpy(),
                            'target_cls':target,'target_img':np.unique(target),
                            'im_name':Path(batch['rgb_path'][k]).name})
            # Metrics ONLY; labels are never accessed by ROI selection.
            mask=batch['rgb_batch_idx']==k
            if bool(mask.any()):
                bb=batch['rgb_semantic_bboxes'][mask]
                gg=bb[bb[:,2:].prod(-1).argmax(),:2]
                e=(ex['center_pred'][k].float()-gg.float()).norm()*max(sem_hw)
                guide_err_sum+=float(e.item()); guide_err_num+=1
            # GTGA q=S*p+t: invert TIR bounding box to RGB-sem and compare IoU.
            tmask=batch['tir_batch_idx']==k
            if int(mask.sum())==1 and int(tmask.sum())==1:
                rb=batch['rgb_semantic_bboxes'][mask][0].float()
                tb=batch['tir_bboxes'][tmask][0].float()
                sc=ex['alignment_info']['sampling_scale'][k].float().reshape(2).clamp_min(1e-6)
                tr=ex['alignment_info']['sampling_translation'][k].float().reshape(2)
                tt=(tb[:2]-tr)/sc;tw=tb[2:]/sc
                l1=rb[:2]-rb[2:]/2;h1=rb[:2]+rb[2:]/2
                l2=tt-tw/2;h2=tt+tw/2
                intersection=(torch.minimum(h1,h2)-torch.maximum(l1,l2)).clamp_min(0).prod()
                union=rb[2:].clamp_min(0).prod()+tw.clamp_min(0).prod()-intersection
                align_iou_sum+=float((intersection/union.clamp_min(1e-7)).item())
                align_center_sum+=float(torch.linalg.norm((tt-rb[:2])*rb.new_tensor([sem_hw[1],sem_hw[0]])).item())
                align_num+=1
        if preview_dir and rank==0 and preview_count<int(ecfg.get('preview_num',8)):
            import cv2
            outdir=Path(preview_dir);outdir.mkdir(parents=True,exist_ok=True)
            for k in range(min(b,int(ecfg.get('preview_num',8))-preview_count)):
                rgb=(ex['rgb_roi'][k].detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
                tir=(ex['tir_roi'][k].detach().cpu().float().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
                guide=(ex['guide_map'][k,0].detach().cpu().float().numpy())
                guide=cv2.resize(guide,(640,640))
                guide=np.uint8(255*(guide-guide.min())/(guide.max()-guide.min()+1e-8))
                guide=cv2.applyColorMap(guide,cv2.COLORMAP_JET)
                tile=np.concatenate([cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),
                                     cv2.cvtColor(tir,cv2.COLOR_RGB2BGR),guide],axis=1)
                cv2.imwrite(str(outdir/f'roi_{preview_count:04d}.jpg'),tile)
                preview_count+=1
        if smoke: break
    pbar.close()
    state={'records':records,'hit':hit,'total':total,'valid':valid,'rois':roi_count,
           'align_iou':align_iou_sum,'align_center':align_center_sum,'align_num':align_num,
           'center_error':guide_err_sum,'center_num':guide_err_num}
    if world>1:
        gathered=[None for _ in range(world)]
        dist.all_gather_object(gathered,state)
    else:
        gathered=[state]
    if rank!=0:return None
    metric=DetMetrics(names={0:'UAV'});n=0
    summed={k:0 for k in ('hit','total','valid','rois','align_iou','align_center',
                           'align_num','center_error','center_num')}
    for item in gathered:
        for key in summed:summed[key]+=item[key]
        for rec in item['records']:
            metric.update_stats(rec)
            n+=1
    result={'images':n,'roi_count':summed['rois'],
            'roi_gt_coverage':summed['hit']/max(summed['total'],1),
            'roi_gt_hit':summed['hit'],'roi_gt_total':summed['total'],
            'tir_valid':summed['valid']/max(summed['rois'],1),
            'guide_center_error_sem_px':summed['center_error']/max(summed['center_num'],1),
            'align_iou':summed['align_iou']/max(summed['align_num'],1),
            'align_center_error_sem_px':summed['align_center']/max(summed['align_num'],1),
            'seconds':time.perf_counter()-val_t0,'metrics':_metric_row(metric)}
    return result


def _checkpoint(path,net,opt,sched,scaler,epoch,best,best_epoch,cfg):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pt.tmp')
    torch.save({
        'architecture':net.architecture,
        'model_state_dict':{k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
        'optimizer_state_dict':opt.state_dict(),
        'scheduler_state_dict':sched.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'epoch':int(epoch),'best_map5095':float(best),'best_epoch':int(best_epoch),'config':cfg,
        'official_yolo_weights':str(net.official_path),
    },tmp)
    os.replace(tmp,path)


def _load_checkpoint(path,net,*,require_opt=False):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('architecture')!=net.architecture:
        raise ValueError('Checkpoint is NOT the single-head variant; do not load old GlobalLocal checkpoints')
    state=obj['model_state_dict']
    if state and all(k.startswith('module.') for k in state):
        state={k[7:]:v for k,v in state.items()}
    net.load_state_dict(state,strict=True)
    if require_opt and ('optimizer_state_dict' not in obj or 'scheduler_state_dict' not in obj):
        raise KeyError('Resume requires optimizer and scheduler')
    return obj


def run_train(args):
    cfg=load_config(args.cfg);tc=cfg['train']
    device_spec=args.device or tc.get('device','5,6')
    if args.smoke:
        device_spec=args.device or '5'
    if int(os.getenv('WORLD_SIZE','1'))>1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl',init_method='env://',timeout=timedelta(hours=3))
    device,rank,_,world=setup_distributed(device_spec)
    set_seed(int(tc.get('seed',0))+rank,bool(tc.get('deterministic',True)))
    batch=int(tc.get('batch',2));accum=int(tc.get('accumulate',16))
    if batch % world or accum<1:
        raise ValueError('Batch must be divisible by ranks; accumulate >=1')
    train_loader,sampler=build_loader(cfg,'train',batch//world,
                                      0 if args.smoke else int(tc.get('workers',4)),rank,world)
    val_loader=None if args.smoke else build_loader(
        cfg,'val',int(tc.get('val_batch',1)),int(tc.get('val_workers',2)),rank,world)[0]
    net=GMLSV2ROISingleHead(cfg,verbose=rank==0).to(device)
    run_dir=Path(cfg['output']['project'])/cfg['output']['name']
    resumed=None
    if args.resume:
        resumed=_load_checkpoint(args.resume,net,require_opt=True)
    elif not args.smoke and (run_dir/'weights'/'last.pt').exists():
        raise FileExistsError(f'Existing checkpoint at {run_dir}/weights/last.pt; '
                              'use --resume or change output.name; no overwrite')
    model=DDP(net,device_ids=[device.index],output_device=device.index,
              find_unused_parameters=True) if world>1 else net
    opt=build_optimizer(net,cfg)
    epochs=int(tc.get('epochs',100))
    sched=build_scheduler(opt,epochs,lrf=float(tc.get('lrf',.01)))
    amp=bool(tc.get('amp',True)) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    first_epoch=0;best=-1.;best_epoch=0
    if resumed is not None:
        opt.load_state_dict(resumed['optimizer_state_dict'])
        sched.load_state_dict(resumed['scheduler_state_dict'])
        if resumed.get('scaler_state_dict'):
            scaler.load_state_dict(resumed['scaler_state_dict'])
        first_epoch=int(resumed['epoch']);best=float(resumed.get('best_map5095',-1))
        best_epoch=int(resumed.get('best_epoch',0))
    if rank==0:
        print(f'[ONE HEAD OFFICIAL PT] world={world} microbatch/rank={batch//world} '
              f'accumulate={accum} effective_batch≈{batch*accum} epochs={epochs} '
              f'resume={bool(resumed)} start_epoch={first_epoch}',flush=True)
        print(f'[PRETRAINED] official YOLO26m.pt={net.official_path}',flush=True)
        (run_dir/'weights').mkdir(parents=True,exist_ok=True)
    for epoch in range(first_epoch,epochs):
        start=time.perf_counter();net.set_epoch(epoch);model.train()
        if sampler is not None:sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        count=0;sum_stats={};lr=float(opt.param_groups[0]['lr'])
        bar=tqdm(train_loader,disable=rank!=0,ncols=154,desc=f'one-head {epoch+1}/{epochs}')
        for step,batch_data in enumerate(bar):
            batch_data=preprocess_batch(batch_data,device)
            sync=((step+1)%accum==0 or (step+1)==len(train_loader))
            group_start=(step//accum)*accum
            denom=min(accum,len(train_loader)-group_start)
            ctx=model.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with ctx:
                with autocast_context(device,amp):
                    loss,parts=model(batch_data)
                    scaled=loss/denom
                if not bool(torch.isfinite(scaled).all()):
                    raise FloatingPointError(f'Non-finite loss epoch={epoch+1} step={step+1}')
                scaler.scale(scaled).backward()
            if sync:
                scaler.unscale_(opt)
                clip=float(tc.get('grad_clip',10.))
                if clip>0:torch.nn.utils.clip_grad_norm_(net.parameters(),clip)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
            count+=1
            for key,val in parts.items():
                sum_stats[key]=sum_stats.get(key,0.)+_asfloat(val)
            if rank==0 and ((step+1)%max(1,int(tc.get('progress_interval',15)))==0 or sync):
                bar.set_postfix({
                    'total':f'{sum_stats.get("total",0)/count:.3f}',
                    'local':f'{sum_stats.get("local_det",0)/count:.3f}',
                    'align':f'{sum_stats.get("align_weighted",0)/count:.3f}',
                    'roi+':f'{sum_stats.get("roi_positive",0)/count:.2f}',
                    'GTcrop':f'{sum_stats.get("gt_assist",0)/count:.2f}',
                })
            if args.smoke:break
        bar.close()
        if args.smoke:
            if rank==0:print('[SMOKE PASS] official YOLO initialized; joint GMLS ROI forward/backward/optimizer step',flush=True)
            break
        # All-rank correct mean of batch metrics (equal num steps from DistributedSampler).
        keys=sorted(sum_stats)
        stats={}
        for key in keys:
            value=torch.tensor([sum_stats[key],count],dtype=torch.float64,device=device)
            if world>1:dist.all_reduce(value,op=dist.ReduceOp.SUM)
            stats[key]=float((value[0]/value[1].clamp_min(1)).item())
        train_seconds=time.perf_counter()-start
        interval=max(1,int(tc.get('val_interval',1)))
        do_val=(epoch==0 or (epoch+1)%interval==0 or epoch+1==epochs)
        report=evaluate(net,val_loader,device,cfg,rank=rank,world=world) if do_val else None
        if rank==0:
            score=report['metrics']['AP50:95'] if report else best
            improved=report is not None and score>best
            if improved:best=score;best_epoch=epoch+1
            print('\n'+'='*104,flush=True)
            print(f'[EPOCH {epoch+1:03d}/{epochs}] LR={lr:.7g} train={train_seconds:.1f}s '
                  f'GPU_mem={torch.cuda.memory_reserved(device)/1024**3:.2f}GiB' if device.type=='cuda' else
                  f'[EPOCH {epoch+1:03d}/{epochs}] LR={lr:.7g} train={train_seconds:.1f}s CPU',flush=True)
            def fmt(*keys):return ' '.join(f'{key}={stats.get(key,float("nan")):.5f}' for key in keys)
            print('[TRAIN LOCAL] '+fmt('local_det','local_box','local_cls','local_dfl'),flush=True)
            print('[TRAIN ALIGN] '+fmt('align_weighted','align_anchor','align_coarse','align_scale','align_targetness'),flush=True)
            print('[GMLS-v2]     '+fmt('align_v2_quality','align_v2_route','align_gmls_gt_quality','align_dmls_gt_quality'),flush=True)
            print('[ROI GUIDE]   '+fmt('roi_guide','roi_center','roi_positive','roi_tir_valid','gt_assist'),flush=True)
            if report:
                m=report['metrics']
                print(f'[VAL SINGLE] P={m["P"]:.5f} R={m["R"]:.5f} AP50={m["AP50"]:.5f} '
                      f'AP75={m["AP75"]:.5f} AP50:95={m["AP50:95"]:.5f}',flush=True)
                print(f'[VAL ROI] GT_coverage={report["roi_gt_coverage"]:.4f} '
                      f'GT_hit={report["roi_gt_hit"]}/{report["roi_gt_total"]} '
                      f'guide_err={report["guide_center_error_sem_px"]:.2f}px '
                      f'TIR_valid={report["tir_valid"]:.3f} RoIs={report["roi_count"]}',flush=True)
                print(f'[VAL GTGA] AlignIoU={report["align_iou"]:.4f} '
                      f'center_error={report["align_center_error_sem_px"]:.2f}px '
                      f'val_seconds={report["seconds"]:.1f}',flush=True)
            else:
                print(f'[VAL] skipped (interval={interval})',flush=True)
            print(f'[BEST] AP50:95={best:.6f} at epoch={best_epoch}',flush=True)
            print('='*104,flush=True)
        sched.step()  # all ranks
        if rank==0:
            if improved:_checkpoint(run_dir/'weights'/'best.pt',net,opt,sched,scaler,epoch+1,best,best_epoch,cfg)
            _checkpoint(run_dir/'weights'/'last.pt',net,opt,sched,scaler,epoch+1,best,best_epoch,cfg)
            row={'epoch':epoch+1,'lr':lr,'train_seconds':train_seconds,'best':best,'best_epoch':best_epoch,**stats}
            if report:
                row.update({f'val_{k}':v for k,v in report['metrics'].items()})
                row.update({f'val_{k}':v for k,v in report.items() if k!='metrics'})
            histfile=run_dir/'history.jsonl'
            with histfile.open('a',encoding='utf-8') as ff:
                ff.write(json.dumps(row,ensure_ascii=False)+'\n')
            # CSV rewritten from JSONL to ensure stable schema and restart support.
            with histfile.open('r',encoding='utf-8') as f:rows=[json.loads(l) for l in f if l.strip()]
            cols=[]
            for r in rows:
                for k in r:
                    if k not in cols:cols.append(k)
            with (run_dir/'epoch_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
                wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(rows)
        if world>1:dist.barrier()
    if world>1 and dist.is_initialized():dist.destroy_process_group()


def run_val(args):
    cfg=load_config(args.cfg)
    from engine.runtime import select_device
    device=select_device(args.device)
    net=GMLSV2ROISingleHead(cfg,verbose=True).to(device)
    ckpt=args.weights or str(Path(cfg['output']['project'])/cfg['output']['name']/'weights'/'best.pt')
    info=_load_checkpoint(ckpt,net)
    print(f'[LOADED] {ckpt}, epoch={info["epoch"]}, architecture={info["architecture"]}',flush=True)
    loader,_=build_loader(cfg,args.split,int(args.batch),int(args.workers),rank=0,world=1)
    report=evaluate(net,loader,device,cfg,smoke=args.smoke,preview_dir=args.plots)
    m=report['metrics']
    print(f'[TEST SINGLE HEAD] P={m["P"]:.6f} R={m["R"]:.6f} AP50={m["AP50"]:.6f} '
          f'AP75={m["AP75"]:.6f} AP50:95={m["AP50:95"]:.6f}',flush=True)
    print(f'[ROI] GT_coverage={report["roi_gt_coverage"]:.4f} '
          f'TIR_valid={report["tir_valid"]:.4f} guide_error={report["guide_center_error_sem_px"]:.2f}px '
          f'AlignIoU={report["align_iou"]:.4f} images={report["images"]}',flush=True)
    target=Path(args.save_dir or (Path(cfg['output']['project'])/cfg['output']['name']/'evaluation'))
    target.mkdir(parents=True,exist_ok=True)
    out=target/f'{args.split}_metrics.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[SAVED] {out}',flush=True)
