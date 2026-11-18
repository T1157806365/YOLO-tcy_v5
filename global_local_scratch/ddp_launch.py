"""Auto-spawn DDP when GMLS-v2 scratch YAML requests multiple devices.

The entrypoint executes this *before* importing the (GPU-heavy) runner.
Manual torchrun remains supported; existing WORLD_SIZE>1 never relaunches.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


def parse_gpu_ids(device_spec):
    if isinstance(device_spec, (tuple, list)):
        parts = [str(x).strip() for x in device_spec]
    else:
        s = str(device_spec).strip().replace('cuda:', '')
        if s.lower() in ('', 'cpu', '-1', 'none'):
            return []
        parts = [x.strip() for x in s.replace(';', ',').split(',')]
    ids = []
    for x in parts:
        if not x:
            continue
        idx = int(x)
        if idx >= 0 and idx not in ids:
            ids.append(idx)
    return ids


def choose_visible_devices(env, requested):
    visible = env.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not visible:
        return ','.join(str(i) for i in requested)
    # Existing CUDA_VISIBLE_DEVICES may use GPU UUIDs rather than integers.
    present = [x.strip() for x in visible.split(',') if x.strip()]
    if len(present) < len(requested):
        raise RuntimeError(
            f'YAML asks for {len(requested)} GPUs {requested}, but '
            f'CUDA_VISIBLE_DEVICES={visible!r} exposes only {len(present)}. '
            'Unset CUDA_VISIBLE_DEVICES or expose the intended GPUs.'
        )
    if len(present) > len(requested):
        # Intentionally make the subset explicit to ensure only required ranks.
        return ','.join(present[:len(requested)])
    return visible


def auto_ddp_launch(args, entrypoint, argv=None, env=None, *, exec_fn=None):
    """May replace current process with python -m torch.distributed.run.

    Returns False for single-card, --smoke, or an existing torchrun child.
    exec_fn injectable for CPU-only tests; normally os.execvpe.
    """
    env = dict(os.environ if env is None else env)
    if int(env.get('WORLD_SIZE', '1')) > 1 or getattr(args, 'smoke', False):
        return False
    with open(args.cfg, encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    spec = args.device if args.device is not None else cfg.get('train', {}).get('device', '0')
    ids = parse_gpu_ids(spec)
    if len(ids) <= 1:
        return False
    visible = choose_visible_devices(env, ids)
    env['CUDA_VISIBLE_DEVICES'] = visible
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('OMP_NUM_THREADS', '1')
    cli = list(sys.argv[1:] if argv is None else argv)
    cmd = [
        sys.executable, '-u', '-m', 'torch.distributed.run',
        '--standalone', f'--nproc_per_node={len(ids)}',
        str(Path(entrypoint).resolve()), *cli,
    ]
    print(f'[AUTO DDP] physical GPUs={visible}; workers={len(ids)}; '
          'torchrun child GPUs become cuda:0/cuda:1.\n'
          f'[AUTO DDP] launching: {" ".join(cmd)}', flush=True)
    (os.execvpe if exec_fn is None else exec_fn)(sys.executable, cmd, env)
    return True  # only reached for injected mock exec_fn
