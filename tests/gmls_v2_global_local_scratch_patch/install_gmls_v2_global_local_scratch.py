#!/usr/bin/env python3
"""Install only NEW files to YOLO-tcy_v5, never touch the original GMLS-v2."""
import argparse
import shutil
from pathlib import Path

FILES = [
    'global_local_scratch/__init__.py',
    'global_local_scratch/geometry.py',
    'global_local_scratch/fusion.py',
    'global_local_scratch/inference.py',
    'global_local_scratch/joint_model.py',
    'global_local_scratch/runner.py',
    'train_gmls_v2_global_local_scratch.py',
    'val_gmls_v2_global_local_scratch.py',
    'configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640_global_local_scratch.yaml',
    'tests/test_gmls_v2_global_local_geometry.py',
    'tests/test_gmls_v2_global_local_mock.py',
]


def install(src: Path, target: Path, *, overwrite: bool = False):
    target=target.resolve()
    original=target/'configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml'
    if not original.is_file():
        raise FileNotFoundError(f'Cannot find original GMLS-v2 config: {original}')
    failures=[]
    for rel in FILES:
        source=src/rel
        dest=target/rel
        if not source.is_file(): raise FileNotFoundError(source)
        if dest.is_file() and dest.read_bytes()!=source.read_bytes() and not overwrite:
            failures.append(str(dest))
    if failures:
        raise FileExistsError('Some new-file paths differ from install package. '
                              'Use --overwrite to update only new files:\n'+'\n'.join(failures))
    for rel in FILES:
        source=src/rel
        dest=target/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or dest.read_bytes()!=source.read_bytes():
            shutil.copy2(source,dest)
        print('[INSTALLED]',rel)
    print('[ORIGINAL UNCHANGED]',original)
    print('[SCRATCH YAML]',target/FILES[8])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True)
    p.add_argument('--overwrite',action='store_true',help='Update only patch-owned paths')
    a=p.parse_args()
    install(Path(__file__).resolve().parent,Path(a.target),overwrite=a.overwrite)
