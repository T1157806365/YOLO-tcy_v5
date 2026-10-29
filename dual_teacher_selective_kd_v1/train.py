"""Dedicated DDP launcher and train entry, no shared source-file modifications."""
from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from engine.runtime import load_config,parse_device_ids
from .factory import validate_cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    path=Path(args.cfg).expanduser().resolve()
    cfg=validate_cfg(load_config(path))
    train=cfg['train'];devices=parse_device_ids(train['device'])
    out=Path(cfg['output']['project'])/cfg['output']['name']
    print(f'[DualTeacherKD] cfg={path}\n  physical_GPUs={devices}  '
          f'patience={train["patience"]}/{train["stage2"]["patience"]}\n  output={out}',flush=True)
    if args.check_only:
        print('[PASS] YAML, local RGB/TIR teacher files and local official student weights found')
        return
    if len(devices)>1 and int(os.environ.get('WORLD_SIZE','1'))==1:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for DDP')
        if (out/'weights'/'stage1_last.pt').exists():
            raise RuntimeError(f'Existing Stage1 checkpoint at {out}; use a new output.name')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(map(str,devices))
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
             f'--nproc_per_node={len(devices)}','-m','dual_teacher_selective_kd_v1.train',
             '--cfg',str(path)]
        print('[DualTeacherKD] CUDA_VISIBLE_DEVICES='+env['CUDA_VISIBLE_DEVICES'],flush=True)
        print('[DualTeacherKD] launching:', ' '.join(cmd),flush=True)
        subprocess.run(cmd,env=env,check=True)
        return
    if (out/'weights'/'stage1_last.pt').exists():
        raise RuntimeError(f'Existing checkpoint at {out}; use a fresh output.name')
    import engine.trainer as trainer
    from .factory import build_model_from_config
    trainer.build_model_from_config=build_model_from_config   # current process only
    trainer.train_experiment(cfg)

if __name__=='__main__':main()
