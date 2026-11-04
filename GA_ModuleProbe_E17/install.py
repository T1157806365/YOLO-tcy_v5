"""Create-only installation into the existing YOLO-tcy_v5 project."""
import argparse
import ast
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',default='.')
    p.add_argument('--device',default='5')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not a.device.isdigit(): raise ValueError('Use one physical GPU number')
    root=Path(a.repo_root).resolve();here=Path(__file__).resolve().parent
    for rel in ('models/gtga_bhlr_factory.py','ga_monoalign_pcgrad/model.py','engine/evaluator.py'):
        if not (root/rel).is_file(): raise FileNotFoundError(root/rel)
    plan=[]
    for src in sorted((here/'ga_module_probe').glob('*.py')):
        ast.parse(src.read_text());plan.append((root/'ga_module_probe'/src.name,src.read_bytes()))
    for label in ('A_support_off','B_support_on'):
        c=yaml.safe_load((here/(label+'.yaml')).read_text())
        w=root/c['source_weights']
        if not w.is_file():raise FileNotFoundError(w)
        c['source_weights']=str(w.resolve());c['tune']['device']=a.device
        c['output']=str(root/c['output'])
        out=Path(c['output'])
        if out.exists() and any(out.iterdir()):raise FileExistsError('Existing experiment: '+str(out))
        plan.append((root/'configs/experiments/ga_module_probe'/(label+'.yaml'),yaml.safe_dump(c,sort_keys=False).encode()))
    for dst,data in plan:
        if dst.exists() and dst.read_bytes()!=data:raise FileExistsError('No overwrite: '+str(dst))
    for dst,data in plan:
        if a.dry_run:print('[CHECK]',dst)
        elif not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            with dst.open('xb') as f:f.write(data)
            print('[CREATED]',dst)
    print('A and B independently load the same source best.pt; do NOT chain B from A.')

if __name__=='__main__':main()
