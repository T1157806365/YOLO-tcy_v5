"""Auto-launch two true torchrun DDP ranks from a familiar `nohup python -u ...`."""
import os,sys
from pathlib import Path
import yaml


def auto_launch(args,script:Path):
    if os.getenv('RANK') is not None or int(os.getenv('WORLD_SIZE','1'))>1:
        return
    cfg=yaml.safe_load(Path(args.cfg).read_text(encoding='utf-8'))
    spec=str(args.device or cfg['train'].get('device','5,6'))
    if spec.lower() in ('cpu','-1','none'):return
    ids=[s.strip() for s in spec.split(',') if s.strip()]
    if len(ids)<=1:return
    if len(ids)!=2 or len(set(ids))!=2:
        raise ValueError(f'Expected two distinct GPUs, got {spec!r}')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',
         '--nproc_per_node=2',str(script),'--cfg',args.cfg]
    if args.resume:cmd += ['--resume',args.resume]
    env=dict(os.environ)
    if env.get('CUDA_VISIBLE_DEVICES') and env['CUDA_VISIBLE_DEVICES']!=spec:
        raise ValueError(f'CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]!r} conflicts with train.device={spec!r}')
    env['CUDA_VISIBLE_DEVICES']=spec
    print(f'[AUTO DDP] physical GPUs={spec}; ranks=2; executable={sys.executable}',flush=True)
    os.execvpe(sys.executable,cmd,env)
