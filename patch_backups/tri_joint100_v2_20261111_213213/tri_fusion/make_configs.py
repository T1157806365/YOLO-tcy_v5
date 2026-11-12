"""Generate four matched 61.4162%-initialized experiment configurations.
All save outside the original stable run; A is an evaluation-only baseline.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import torch
import yaml

VARIANTS = {
    'A': ('identity', False, None),
    'B': ('RGB_TIR', True, 'rgb_tir'),
    'C': ('RGB_TIR_HR', True, 'rgb_tir_hr'),
    'D': ('RGB_TIR_HR_ALIGN', True, 'align_rgb_tir_hr'),
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-weights',required=True)
    p.add_argument('--out',default='configs/experiments/tri_fusion_614162')
    p.add_argument('--device',default='3')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--accumulate',type=int,default=8)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=0.0002)
    args=p.parse_args()
    if int(args.epochs)<1 or int(args.accumulate)<1:
        raise ValueError('epochs and accumulate must be positive')
    path=Path(args.base_weights).expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(path)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    base=ck['config']
    mod=base['model']
    if (mod.get('variant')!='rlsfa_gtga_centered_bhlr_v2' or
        mod.get('fusion')!='add' or
        mod.get('bhlr',{}).get('injection_position')!='post_fusion' or
        not mod.get('bhlr',{}).get('enabled',False) or
        not mod.get('rlsfa_gtga_centered',{}).get('enabled',False)):
        raise ValueError('Checkpoint is NOT the full stable GTGA + Post-Fusion BHLR baseline')
    if any('aghr' in k.lower() or 'accf' in k.lower() or 'tri_gate' in k.lower()
           for k in ck['model_state_dict']):
        raise ValueError('Base checkpoint contains a newer module; wrong input weights')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for variant,(name,enabled,mode) in VARIANTS.items():
        conf=copy.deepcopy(base)
        conf['model']['tri_gate']={
            'enabled': bool(enabled),
            'mode': mode or 'rgb_tir',
            'hidden':24,
            'max_change':1.0,
        }
        # Training settings for isolated 20e *module-only* experiment.
        conf['tri_experiment']={
            'variant':variant, 'base_weights':str(path),
            'epochs':int(args.epochs), 'device':str(args.device),
            'batch':int(args.batch), 'accumulate':int(args.accumulate),
            'lr':float(args.lr), 'weight_decay':0.0001,
            'identity_reg':0.00005,
            'workers':4, 'seed':0,
            'amp':True, 'grad_clip':5.0,
        }
        conf['output']={'project':'runs/tri_fusion_614162',
                        'name':f'{variant}_{name}_seed0'}
        dest=out/f'{variant}_{name}.yaml'
        if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
        dest.write_text(yaml.safe_dump(conf,allow_unicode=True,sort_keys=False),encoding='utf-8')
        print('[CREATED]',dest)
    print('A: validation only. B/C/D: train tri-gate only from SAME complete 61.4162% model.')

if __name__=='__main__':main()
