"""Same-weight validation with local HR injection ON and OFF."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='val',choices=['val','test'])
    p.add_argument('--device',default='3')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=0.001)
    p.add_argument('--iou',type=float,default=0.7)
    args=p.parse_args()
    from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
    from models.gtga_bhlr_factory import load_model_from_checkpoint
    from engine.evaluator import validate_metrics
    device=torch.device('cuda:'+args.device if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    model,ck,cfg=load_model_from_checkpoint(args.weights,device=device,strict=True,verbose=False)
    if not hasattr(model,'ag_lhfe'):
        raise RuntimeError('Only AG-LHFE checkpoints are accepted')
    d=cfg['data']
    ds=build_dual_resolution_dataset(d['rgb'],d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(ds,batch_size=args.batch,workers=args.workers,
                                             shuffle=False,pin_memory=device.type=='cuda')
    metrics={}
    for enabled in (True,False):
        model.ag_lhfe.injection_enabled=enabled
        name='on' if enabled else 'off'
        print(f'==== AG-LHFE {name} ====')
        metrics[name]=validate_metrics(model,loader,device,amp=device.type=='cuda',
            conf_thres=args.conf,iou_thres=args.iou,max_det=300,preview_num=0)
    diff={k:metrics['on'][k]-metrics['off'][k]
          for k in ('map50','map75','map5095','precision','recall')}
    result={'weights':args.weights,'split':args.split,'on':metrics['on'],'off':metrics['off'],'on_minus_off':diff}
    output=Path(args.weights).parent.parent/f'local_hr_{args.split}_on_off.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('Saved:',output)
if __name__=='__main__':main()
