"""Export EXACT original student core for unchanged GMLS-v2 alignment evaluator.

The exported model excludes training-only RGB/TIR auxiliary heads. It will
load strictly in original models.gtga_bhlr_factory.load_model_from_checkpoint.
"""
from __future__ import annotations
import argparse,torch
from pathlib import Path
from models.gtga_bhlr_factory import load_model_from_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    dest=Path(args.output)
    if dest.exists():raise FileExistsError(dest)
    ck=torch.load(args.weights,map_location='cpu',weights_only=False)
    sd=ck['model_state_dict']
    if sd and all(k.startswith('module.') for k in sd):sd={k[7:]:v for k,v in sd.items()}
    core={k[len('core.'):]:v for k,v in sd.items() if k.startswith('core.')}
    if not core:raise RuntimeError('No core.* keys, not a MonoAlign student checkpoint')
    out={k:v for k,v in ck.items() if k not in ('model_state_dict','optimizer_state_dict','scheduler_state_dict','scaler_state_dict')}
    out['model_state_dict']=core
    out['config']=dict(ck['config'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,dest)
    model,_,_=load_model_from_checkpoint(dest,device='cpu',strict=True,verbose=False)
    del model
    print('[PASS] Original GMLS-v2 checkpoint strictly loadable:',dest)

if __name__=='__main__':main()
