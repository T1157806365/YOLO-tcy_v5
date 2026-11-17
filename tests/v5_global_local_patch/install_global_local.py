#!/usr/bin/env python3
"""Install only new files into existing YOLO-tcy_v5, backing up conflicts.

Does NOT edit any preexisting v5 Python/YAML/checkpoint code.
"""
from __future__ import annotations
import argparse, shutil, datetime
from pathlib import Path


def install(source:Path,target:Path):
    if not (target/'models/gtga_bhlr_factory.py').is_file():
        raise FileNotFoundError(f'Not the expected YOLO-tcy_v5 checkout: {target}')
    if not (target/'engine/trainer.py').is_file():
        raise FileNotFoundError(f'Missing v5 trainer: {target}')
    backup=target/'.global_local_install_backup'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    copied=[]
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name.startswith('.') or '__pycache__' in path.parts or path.suffix=='.pyc':
            continue
        rel=path.relative_to(source)
        if rel.name in ('install_global_local.py','README_使用说明.md'):continue
        to=target/rel
        if to.is_file():
            old=to.read_bytes();new=path.read_bytes()
            if old==new:continue
            b=backup/rel;b.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(to,b)
        to.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,to)
        copied.append(str(rel))
    print('Installed new global-local files into:',target)
    for p in copied:print('  +',p)
    print('Original GTGA/BHLR/HVAD/engine files were not modified.')
    if backup.exists():print('Backed up previous patch file versions under:',backup)
    return copied


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True)
    a=p.parse_args()
    install(Path(__file__).resolve().parent,Path(a.target).expanduser().resolve())
if __name__=='__main__':main()
