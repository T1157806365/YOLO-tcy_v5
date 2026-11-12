"""Generate matched 100-epoch independent RGB-T students, never load an old student .pt.

The base is the original Post-Fusion BHLR *YAML*, not a checkpoint. Original
YOLO initialization and dual SINGLE-modal teacher initialization are retained.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import yaml

ARMS = {
    'A': ('baseline', False, 'rgb_tir'),
    'B': ('rgb_tir', True, 'rgb_tir'),
    'C': ('rgb_tir_hr', True, 'rgb_tir_hr'),
    'D': ('rgb_tir_hr_align', True, 'align_rgb_tir_hr'),
}


def generate(base: dict, out: Path, device='2,3', global_batch=32,
             accumulate=1, epochs=100, workers=4, seed=0):
    if epochs != 100:
        raise ValueError('Single100 study requires exactly 100 epochs')
    if global_batch < 1 or accumulate < 1:
        raise ValueError('batch and accumulate must be positive')
    if any(k in base for k in ('tri_experiment', 'accf_experiment', 'aghr_experiment')):
        raise ValueError('Base YAML contains prior experiment-specific initialization')
    m = base['model']
    if not (m.get('variant') == 'rlsfa_gtga_centered_bhlr_v2' and
            m.get('fusion') == 'add' and
            m.get('bhlr', {}).get('injection_position') == 'post_fusion' and
            m.get('bhlr', {}).get('enabled') is True and
            m.get('rlsfa_gtga_centered', {}).get('enabled') is True):
        raise ValueError('Expected stable RLSFA-GTGA + post_fusion BHLR YAML')
    if base['monoalign'].get('init_from_teachers') is not True:
        raise ValueError('Require two SINGLE-modal teacher initializations')
    if base['monoalign'].get('schedule_mode') != 'single100_step_v1':
        raise ValueError('Require original single100_step_v1 Mono/KD schedule')
    if base['monoalign'].get('pcgrad_enabled') is not True:
        raise ValueError('Require original PCGrad')
    if str(base['train'].get('optimizer')).lower() != 'sgd':
        raise ValueError('Keep SGD of stable baseline')
    if base['train'].get('resume') is not None or base['train'].get('init_student_weights') is not None:
        raise ValueError('Old student checkpoint initialization must be null')
    out = Path(out)
    targets = {k: out / f'{k}_{name}_single100.yaml' for k,(name,_,_) in ARMS.items()}
    collisions = [p for p in targets.values() if p.exists()]
    if collisions:
        raise FileExistsError('Refusing to overwrite existing YAML: '+', '.join(map(str, collisions)))
    result = {}
    for arm, (name, enabled, mode) in ARMS.items():
        cfg = copy.deepcopy(base)
        cfg['model']['tri_gate'] = {
            'enabled': enabled,
            'mode': mode,
            'hidden': 24,
            'max_change': 1.0,
        }
        tr=cfg['train']
        tr.update({
            'device': str(device),
            'batch_mode': 'global',
            'batch': int(global_batch),
            'accumulate': int(accumulate),
            'epochs': 100,
            'workers': int(workers),
            'seed': int(seed),
            'resume': None,
            'init_student_weights': None,
            'patience': 100,
        })
        tr.setdefault('stage2', {})['enabled'] = False
        tr['stage2']['epochs'] = 0
        cfg['monoalign']['minimum_stage1_epochs'] = 100
        cfg['monoalign']['init_from_teachers'] = True
        cfg['monoalign']['pcgrad_enabled'] = True
        cfg['monoalign']['schedule_mode'] = 'single100_step_v1'
        cfg['output'] = {'project':'runs/tri_fusion_joint100',
                         'name':f'yolo26m_UAVCB_{arm}_{name}_seed{seed}'}
        if 'tri_experiment' in cfg:
            del cfg['tri_experiment']
        result[targets[arm]] = cfg
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base', default=str(Path(__file__).resolve().parent/'templates'/'stable_postfusion_single100.yaml'),
                   help='Historical single100 POST-FUSION YAML (NOT old student checkpoint)')
    p.add_argument('--out',default='configs/experiments/tri_fusion_joint100')
    p.add_argument('--device',default='2,3',help='Physical GPUs: 2,3 or a single GPU e.g. 3')
    p.add_argument('--global-batch',type=int,default=32)
    p.add_argument('--accumulate',type=int,default=1)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    base_path=Path(args.base)
    if base_path.suffix.lower() not in ('.yaml','.yml'):
        raise ValueError('--base must be YAML; this study never reads old RGB-T student .pt')
    base=yaml.safe_load(base_path.read_text(encoding='utf-8'))
    generated=generate(base, Path(args.out),device=args.device,global_batch=args.global_batch,
                       accumulate=args.accumulate,epochs=args.epochs,
                       workers=args.workers,seed=args.seed)
    for path in generated:
        path.parent.mkdir(parents=True,exist_ok=True)
    for path, config in generated.items():
        path.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
        print('[CREATED]',path)
    print('[SINGLE STAGE] 100 epochs, stage2 off; full student trainable; base student .pt NOT used')
    print('[INITIALIZATION] Official YOLO weights + RGB/TIR single-modal teacher init')
    print('[LOSSES] Original detection + alignment + Mono + ROI-KD + PCGrad unchanged')
    print(f'[BATCH] global micro-batch={args.global_batch}, accumulate={args.accumulate}, '
          f'effective={args.global_batch*args.accumulate}')

if __name__=='__main__':
    main()
