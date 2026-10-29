#!/usr/bin/env python3
"""Conservative installer/restorer. Never edits existing repo source code."""
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

MANIFEST = '.rgb_preserve_v1_install.json'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def files_in(package):
    for rel_dir in ('rgb_preserve_v1', 'configs/experiments/rgb_preserve_v1'):
        for p in (package / rel_dir).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                yield p.relative_to(package)

def root_check(project):
    for path in ('engine/trainer.py', 'models/gtga_bhlr_factory.py', 'val_rlsfa_gtga_bhlr_v2.py'):
        if not (project / path).is_file():
            raise SystemExit('Not YOLO-tcy_v5 project root: missing ' + path)

def install(package, project):
    root_check(project)
    manifest_path = project / MANIFEST
    if manifest_path.exists():
        raise SystemExit('Already installed. Restore first if you want a fresh install.')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_root = project / '.rgb_preserve_v1_recovery' / stamp
    records=[]
    for rel in sorted(files_in(package)):
        src, dst=package/rel, project/rel
        backup=None
        if dst.exists():
            backup=backup_root/rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        records.append({'path':str(rel), 'installed_sha256':sha(dst), 'backup':str(backup.relative_to(project)) if backup else None})
    manifest_path.write_text(json.dumps({'version':'1.0','records':records,'backup_root':str(backup_root.relative_to(project))},indent=2),encoding='utf-8')
    print('[INSTALL OK] Added files:', len(records))
    print('[INSTALL OK] Existing core code not touched.')
    print('Restore anytime: bash rgb_preserve_v1/restore.sh')
    print('Before training: python rgb_preserve_v1/check_identity.py --cfg configs/experiments/rgb_preserve_v1/uavcb_rgb_preserve_640.yaml --device 3')

def restore(project):
    root_check(project)
    mp=project/MANIFEST
    if not mp.exists():
        raise SystemExit('No install manifest; nothing to restore')
    manifest=json.loads(mp.read_text())
    safety_dir=project/manifest['backup_root']/'user_modified_at_restore'
    for rec in reversed(manifest['records']):
        path=project/rec['path']
        if path.exists():
            if sha(path)!=rec['installed_sha256']:
                moved=safety_dir/rec['path']
                moved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path,moved)
                print('[SAFE COPY] User-edited file kept at', moved)
            path.unlink()
        if rec['backup']:
            b=project/rec['backup']
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(b,path)
    mp.unlink()
    print('[RESTORE OK] Restored files that existed before install; removed this experiment source files.')
    print('[RESTORE OK] Training runs and log outputs were not touched.')

if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[1] not in ('install','restore'):
        raise SystemExit('Usage: python manage_install.py install|restore <package_dir> <project_root>')
    op,pkg,proj=sys.argv[1],Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve()
    (install(pkg,proj) if op=='install' else restore(proj))
