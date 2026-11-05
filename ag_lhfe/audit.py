"""Report validation candidate coverage and injection statistics of AG-LHFE.

Does not use GT to select patches; GT only computes diagnostic coverage.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--device',default='3')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--images',type=int,default=96,help='0 = all')
    p.add_argument('--save-dir',default='runs/ag_lhfe_audit')
    args=p.parse_args()
    from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
    from engine.runtime import preprocess_batch,autocast_context
    from models.gtga_bhlr_factory import load_model_from_checkpoint
    device=torch.device('cuda:'+args.device if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    model,ck,cfg=load_model_from_checkpoint(args.weights,device=device,strict=True,verbose=False)
    if not hasattr(model,'ag_lhfe'):
        raise RuntimeError('Checkpoint is not an AG-LHFE model')
    d=cfg['data']
    ds=build_dual_resolution_dataset(d['rgb'],d['tir'],split=args.split,
        rgb_high_imgsz=int(d.get('rgb_imgsz',1920)),rgb_semantic_imgsz=int(d.get('rgb_semantic_imgsz',640)),
        tir_imgsz=int(d.get('tir_imgsz',640)),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=args.batch,workers=args.workers,
                                             shuffle=False,pin_memory=device.type=='cuda')
    total_gt=0;covered=0;images=0;ratios=[];obj=[];offset=[];by_type=[]
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader,desc='AG-LHFE candidate audit'):
            batch=preprocess_batch(batch,device)
            with autocast_context(device,device.type=='cuda'):
                pred=model.predict(batch['rgb_img'],batch['tir_img'],rgb_semantic=batch['rgb_semantic_img'],
                                   return_features=True,return_alignment_debug=True)
            diag=pred['ag_lhfe_debug']
            centers=diag['centers_p3']
            b,k,_=centers.shape
            H,W=pred['rgb_backbone_raw'][model.ag_lhfe_p3_layer].shape[-2:]
            ixs=batch['rgb_batch_idx'].view(-1).long()
            for bi in range(b):
                if args.images and images>=args.images:break
                boxes=batch['rgb_semantic_bboxes'][ixs==bi]
                xy=boxes[:,:2]*boxes.new_tensor([W,H])-0.5
                if len(xy):
                    dist=torch.cdist(centers[bi].float(),xy.float())
                    covered+=int((dist.min(0).values<=model.ag_lhfe.positive_radius).sum().item())
                    total_gt+=len(xy)
                images+=1
                ratios.append(float(diag['injection_rms_ratio'].float().item()))
                obj.append(float(diag['roi_objectness'][bi].float().mean().item()))
                offset.append(float(diag['local_offsets'][bi].float().abs().mean().item()))
                by_type.append(float(diag['proposal_kind'][bi].float().mean().item()))
            if args.images and images>=args.images:break
    result={'checkpoint':str(Path(args.weights).resolve()),'split':args.split,'images':images,
        'gt':total_gt,'proposal_recall':covered/max(total_gt,1),
        'mean_injection_rms_ratio':sum(ratios)/max(1,len(ratios)),
        'mean_roi_objectness':sum(obj)/max(1,len(obj)),
        'mean_absolute_offset_p3':sum(offset)/max(1,len(offset)),
        'fraction_crossmodal_proposals':sum(by_type)/max(1,len(by_type)),
        'strength':float((model.ag_lhfe.strength_max*model.ag_lhfe.strength_logit.sigmoid()).item())}
    out=Path(args.save_dir)/Path(args.weights).parent.parent.name/args.split
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));print('Saved:',out/'summary.json')
if __name__=='__main__':main()
