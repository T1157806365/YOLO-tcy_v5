#!/usr/bin/env python3
"""Export predicted-center native RGB/TIR crops for a separate local detector.

The original UAV-CB images and labels are never changed. No GT coordinates are
used for crop proposals, on train/val/test. Labels only supervise resulting ROIs.

Usage: python export_global_local_rois.py --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml --splits train val
"""
from __future__ import annotations

import argparse, hashlib, json, random
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset, build_dual_resolution_dataloader
from engine.runtime import load_config, preprocess_batch, select_device
from models.gtga_bhlr_factory import load_model_from_checkpoint
from global_local.inference import generate_rois, get_global_results
from global_local.geometry import local_gt_from_high, sample_native_hr_and_tir, _ratio_pad


def save_roi(rgb, tir, rgb_file: Path, tir_file: Path):
    """Tensors are RGB + 3-channel repeated grayscale TIR, normalized [0,1]."""
    rgb_file.parent.mkdir(parents=True,exist_ok=True)
    tir_file.parent.mkdir(parents=True,exist_ok=True)
    r=(rgb.detach().float().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy())
    t=(tir[0].detach().float().clamp(0,1).mul(255).round().byte().cpu().numpy())
    if not cv2.imwrite(str(rgb_file),cv2.cvtColor(r,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,96]):
        raise IOError(f'Could not write {rgb_file}')
    if not cv2.imwrite(str(tir_file),t,[cv2.IMWRITE_JPEG_QUALITY,98]):
        raise IOError(f'Could not write {tir_file}')


def export_split(cfg, model, device, split, max_images=0, extra_jitter=0, seed=0):
    data=cfg['data']; roi=cfg['roi']
    root=Path(data['roi_root']).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    ds=build_dual_resolution_dataset(
        rgb_yaml=data['rgb'],tir_yaml=data['tir'],split=split,
        rgb_high_imgsz=int(data.get('rgb_imgsz',1920)),
        rgb_semantic_imgsz=int(data.get('rgb_semantic_imgsz',640)),
        tir_imgsz=int(data.get('tir_imgsz',640)),
        pair_mode=str(data.get('pair_mode','relative')),
        augment=False,fliplr=0.0,flipud=0.0,tir_channels=3,strict_pair=True,
    )
    loader=build_dual_resolution_dataloader(ds,batch_size=1,workers=int(cfg.get('export',{}).get('workers',2)),shuffle=False,pin_memory=True,drop_last=False)
    gen=random.Random(seed)
    max_rois=int(roi.get('topk',2)); size=int(roi.get('size',640))
    min_visible=float(roi.get('min_visible',0.60))
    min_score=float(roi.get('candidate_conf',0.001))
    manifest=root/f'{split}.jsonl'
    records=0; positive=0; nimgs=0; samples_with_roi_gt=0; valid_frac=[]
    print(f'Exporting {split} -> {manifest}, topk={max_rois}, roi={size}')
    with manifest.open('w',encoding='utf-8') as fh:
        for batch in tqdm(loader,total=(min(len(loader),max_images) if max_images else len(loader)),desc=f'ROI {split}'):
            if max_images and nimgs>=max_images:break
            nimgs+=1
            batch=preprocess_batch(batch,device)
            with torch.inference_mode():
                global_predictions,info=get_global_results(model,batch)
                roi_groups,_=generate_rois(
                    model,batch,topk=max_rois,roi_size=size,
                    min_score=min_score,min_center_dist=float(roi.get('min_center_dist',32.0)),
                    predictions=global_predictions,alignment_info=info,
                )
            high=batch['rgb_img'][0]; tir_full=batch['tir_img'][0]
            _,H,W=high.shape
            sem_hw=tuple(batch['rgb_semantic_img'].shape[-2:])
            all_gt=batch['rgb_bboxes'][batch['rgb_batch_idx']==0].detach()
            pair_key=batch['pair_key'][0]
            src_path=batch['rgb_path'][0]
            is_positive=False
            for p in roi_groups[0]:
                variants=[(p.rgb,p.tir,p.geometry,p.center_sem_xy,'base')]
                if split=='train':
                    rh,_,_=_ratio_pad(batch['rgb_ratio_pad'][0]);rs,_,_=_ratio_pad(batch['rgb_semantic_ratio_pad'][0])
                    for aug_i in range(int(extra_jitter)):
                        dx=gen.uniform(-float(roi.get('jitter_high_px',48)),float(roi.get('jitter_high_px',48)))*rs/rh
                        dy=gen.uniform(-float(roi.get('jitter_high_px',48)),float(roi.get('jitter_high_px',48)))*rs/rh
                        center=(p.center_sem_xy[0]+dx,p.center_sem_xy[1]+dy)
                        with torch.inference_mode():
                            # Reuse the exact coarse geometry from the global model,
                            # obtained already in generate_rois (stored temporarily below).
                            rr,tt,gg=sample_native_hr_and_tir(
                                high,tir_full,center,sem_hw,
                                batch['rgb_ratio_pad'][0],batch['rgb_semantic_ratio_pad'][0],
                                info["sampling_scale"][0],info["sampling_translation"][0],roi_size=size,
                            )
                        variants.append((rr,tt,gg,center,f'jitter{aug_i}'))
                for rc,tc,geom,center,variant in variants:
                    boxes=local_gt_from_high(all_gt,geom.x0,geom.y0,size,W,H,min_visible=min_visible)
                    if len(boxes)>0:
                        is_positive=True;positive+=1
                    idx=hashlib.sha1(f'{split}|{pair_key}|{p.rank}|{variant}'.encode('utf-8')).hexdigest()[:20]
                    rgb_rel=Path('images')/split/'RGB'/f'{idx}.jpg'
                    tir_rel=Path('images')/split/'TIR'/f'{idx}.jpg'
                    save_roi(rc,tc,root/rgb_rel,root/tir_rel)
                    label_rel=Path('labels')/split/f'{idx}.txt'
                    (root/label_rel).parent.mkdir(parents=True,exist_ok=True)
                    (root/label_rel).write_text(''.join('0 '+' '.join(f'{v:.7f}' for v in row)+'\n' for row in boxes.cpu().tolist()),encoding='utf-8')
                    entry={
                        'id':idx,'split':split,'source_pair':pair_key,'source_rgb':src_path,
                        'rgb_path':rgb_rel.as_posix(),'tir_path':tir_rel.as_posix(),
                        'label_path':label_rel.as_posix(),
                        'boxes_xywh':boxes.cpu().tolist(),
                        'candidate_rank':p.rank,'candidate_confidence':p.score,
                        'crop_xy':(geom.x0,geom.y0),'crop_size':geom.size,
                        'center_sem_xy':center,'tir_valid_fraction':geom.tir_valid_fraction,
                        'high_hw':(H,W),'sem_hw':sem_hw,'from_detector':p.from_detector,
                    }
                    fh.write(json.dumps(entry,ensure_ascii=False)+'\n')
                    records+=1;valid_frac.append(geom.tir_valid_fraction)
            samples_with_roi_gt+=int(is_positive)
    summary={'split':split,'images':nimgs,'rois':records,'positive_rois':positive,
             'images_with_gt_in_at_least_one_roi':samples_with_roi_gt,
             'mean_tir_valid_fraction':sum(valid_frac)/max(len(valid_frac),1),
             'manifest':str(manifest)}
    (root/f'{split}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--splits',nargs='+',choices=('train','val','test'),default=['train','val'])
    p.add_argument('--device',default=None)
    p.add_argument('--max-images',type=int,default=0)
    args=p.parse_args()
    cfg=load_config(args.cfg)
    device=select_device(args.device or cfg.get('export',{}).get('device','5'))
    path=cfg['global']['weights']
    model,_,_=load_model_from_checkpoint(path,device=device,strict=True,verbose=False)
    model.eval()
    for name in args.splits:
        export_split(cfg,model,device,name,max_images=args.max_images,
                     extra_jitter=int(cfg.get('roi',{}).get('train_extra_jitter',0)),
                     seed=int(cfg.get('train',{}).get('seed',0)))

if __name__=='__main__':
    main()
