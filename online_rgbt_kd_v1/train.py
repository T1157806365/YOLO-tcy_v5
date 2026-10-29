"""python -m online_rgbt_kd_v1.train --cfg configs/.../uavcb_online_kd_1920.yaml"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))

from engine.runtime import load_config, parse_device_ids
from online_rgbt_kd_v1.factory import validate_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg_path = Path(args.cfg).expanduser().resolve()
    cfg = validate_cfg(load_config(cfg_path))
    output = Path(cfg['output']['project']) / str(cfg['output']['name'])
    print(f'[OnlineKD] config={cfg_path}; out={output}; patience={cfg["train"]["patience"]}/'
          f'{cfg["train"]["stage2"]["patience"]}', flush=True)
    if args.check_only:
        print('[ONLINE KD PREFLIGHT] YAML and local official checkpoint paths OK', flush=True)
        return
    devices = parse_device_ids(cfg['train']['device'])
    in_ddp = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if len(devices) > 1 and not in_ddp:
        if not torch_cuda_available():
            raise RuntimeError('Multi-GPU requested, but CUDA is unavailable')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={len(devices)}', '-m', 'online_rgbt_kd_v1.train',
               '--cfg', str(cfg_path)]
        print('Launching DDP:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        return
    if not in_ddp and (output / 'weights' / 'stage1_last.pt').exists():
        raise RuntimeError(f'Experiment contains checkpoints: {output}. Use a new output.name; do not overwrite.')
    # Keep the original training code on disk unchanged; inject new factory only
    # into this Python process. DDP child executes the same entrypoint.
    import engine.trainer as trainer
    from online_rgbt_kd_v1.factory import build_model_from_config
    trainer.build_model_from_config = build_model_from_config
    trainer.train_experiment(cfg)


def torch_cuda_available():
    import torch
    return torch.cuda.is_available()

if __name__ == '__main__': main()
