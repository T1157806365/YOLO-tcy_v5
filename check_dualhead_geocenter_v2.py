#!/usr/bin/env python3
"""Read-only audit of installed GMLS-v2 GeoCenter dual-head implementation.
Run in YOLO-tcy_v5 project root: python check_dualhead_geocenter_v2.py
No model weights are loaded and no files are modified.
"""
from pathlib import Path
import ast
import json
import re
import sys

try:
    import yaml
except ImportError:
    print('[ERROR] PyYAML missing; run in your tcy conda environment')
    sys.exit(2)

root = Path.cwd()
base = Path('configs/experiments/gmls_v2_roi')
files = {
    'V2 parent model': Path('gmls_roi_onehead_v2/model.py'),
    'V2 parent runner': Path('gmls_roi_onehead_v2/runner.py'),
    'V2 center formula': Path('gmls_roi_onehead_v2/geo_center.py'),
    'Dual model': Path('gmls_roi_twohead_v1/model.py'),
    'Dual runner': Path('gmls_roi_twohead_v1/runner.py'),
    'Dual benchmark': Path('gmls_roi_twohead_v1/benchmark.py'),
    'Dual config': base / 'uavcb_gmls_v2_geo_center_dualhead_scratch100.yaml',
    'V2 config': base / 'uavcb_gmls_v2_hybrid_geo_center_scratch100_v2.yaml',
    'Install manifest': Path('gmls_roi_twohead_v1/install_manifest.json'),
}
print('=' * 72)
print('READ-ONLY GeoCenter V2 + DualHead audit')
print('PROJECT ROOT:', root)
print('=' * 72)
for key, path in files.items():
    print(f"{'FOUND' if (root / path).is_file() else 'MISSING':7} | {key:20} | {path}")

def read(key):
    path = root / files[key]
    return path.read_text(encoding='utf-8') if path.is_file() else ''

def check(label, condition):
    status = 'PASS' if condition else 'FAIL'
    print(f'[{status}] {label}')
    return bool(condition)

print('\n=== A. Source code (not just config) ===')
p = read('V2 parent model')
r = read('V2 parent runner')
g = read('V2 center formula')
d = read('Dual model')
dr = read('Dual runner')
checks = {}
checks['v2_hook'] = check('V2 model imports and invokes final_center_loss',
    'from .geo_center import final_center_loss' in p and
    bool(re.search(r'(?<!def )\bfinal_center_loss\s*\(', p)))
checks['v2_loss'] = check('V2 final-center loss joins alignment_loss',
    'final_center_weight' in p and
    'alignment_loss=alignment_loss+final_center_weight*geo_ramp' in p.replace(' ', ''))
checks['v2_ramp'] = check('V2 warmup calculation and metric are present',
    'final_center_warmup_epochs' in p and 'align_final_center_ramp' in p)
checks['v2_formula'] = check('GeoCenter formula does not detach predicted center',
    'sampling_scale' in g and 'sampling_translation' in g and
    'center_pred = (t[0, :2] - trans[i]) / scale[i]' in g and
    'loss_i = F.smooth_l1_loss' in g)
checks['v2_runner'] = check('V2 runner logs final-center metrics',
    'align_final_center' in r and 'best_align.pt' in r)
checks['dual_inherit'] = check('DualHead inherits installed V2 parent',
    'from gmls_roi_onehead_v2.model import GMLSV2OneHeadHybrid' in d and
    'class GMLSV2DualHeadHybrid(GMLSV2OneHeadHybrid)' in d)
checks['dual_indep'] = check('Independent global Neck+Detect copied',
    'global_neck_detect' in d and 'deepcopy(layers[n:])' in d)
checks['dual_hook'] = check('DualHead overrides parent Global prediction hook',
    'def _shared_head_global_prediction(' in d and
    '_shared_head_global_prediction(' in p)
checks['dual_runner'] = check('Dual runner uses DualHead class',
    'GMLSV2DualHeadHybrid' in dr)
checks['dual_val'] = check('Dual benchmark uses DualHead model',
    'GMLSV2DualHeadHybrid' in read('Dual benchmark'))
for key in ('V2 parent model','V2 parent runner','Dual model','Dual runner'):
    s = read(key)
    if s:
        try:
            ast.parse(s)
            print(f'[PASS] Syntax parses: {key}')
        except SyntaxError as e:
            print(f'[FAIL] Syntax: {key}: {e}')

print('\n=== B. Config in the actual dual training path ===')
try:
    cfg = yaml.safe_load(read('Dual config')) if read('Dual config') else None
except yaml.YAMLError as e:
    print('[FAIL] Config parsing:', e)
    cfg = None

def get(cfg, *keys):
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            return '<MISSING>'
        v = v[k]
    return v

important = [
    ('model.name',('model','name')),
    ('model.pretrained',('model','pretrained')),
    ('model.dual_head.enabled',('model','dual_head','enabled')),
    ('model.roi_single_head.topk',('model','roi_single_head','topk')),
    ('model.roi_single_head.top2_diverse_enabled',('model','roi_single_head','top2_diverse_enabled')),
    ('loss.alignment.enabled',('loss','alignment','enabled')),
    ('loss.alignment.lambda_final_center',('loss','alignment','lambda_final_center')),
    ('loss.alignment.final_center_warmup_epochs',('loss','alignment','final_center_warmup_epochs')),
    ('loss.roi.lambda_global_aux',('loss','roi','lambda_global_aux')),
    ('train.epochs',('train','epochs')),
    ('train.batch',('train','batch')),
    ('train.accumulate',('train','accumulate')),
    ('train.device',('train','device')),
    ('train.resume',('train','resume')),
    ('output.name',('output','name')),
]
if cfg:
    for name, path in important:
        print(f'{name:52} = {get(cfg,*path)!r}')
    checks['cfg_v2'] = check('Dual YAML enables V2 0.25 + warmup 10',
        get(cfg,'loss','alignment','lambda_final_center') == 0.25 and
        get(cfg,'loss','alignment','final_center_warmup_epochs') == 10)
    checks['cfg_top2'] = check('Dual YAML has Top2',
        get(cfg,'model','roi_single_head','topk') == 2)
    checks['cfg_global'] = check('Dual YAML enables global auxiliary 0.5',
        get(cfg,'loss','roi','lambda_global_aux') == 0.5)
    checks['cfg_dual'] = check('Dual YAML enables independent DualHead',
        get(cfg,'model','dual_head','enabled') is True)
    print('\nAdditional *quality*, *route*, *guide*, *center*, *diverse* key paths:')
    def walk(obj, prefix=''):
        if isinstance(obj, dict):
            for k,v in obj.items():
                key = f'{prefix}.{k}' if prefix else str(k)
                if any(term in str(k).lower() for term in ('quality','route','guide','center','diverse')):
                    if not isinstance(v, (dict,list)):
                        print(' ',key,'=',repr(v))
                walk(v,key)
        elif isinstance(obj,list):
            for i,v in enumerate(obj): walk(v,f'{prefix}[{i}]')
    walk(cfg)
    def find_leaf(node, leaf, path=''):
        found = []
        if isinstance(node, dict):
            for k,v in node.items():
                cur = f'{path}.{k}' if path else str(k)
                if k == leaf:
                    found.append((cur,v))
                found.extend(find_leaf(v,leaf,cur))
        elif isinstance(node,list):
            for i,v in enumerate(node):
                found.extend(find_leaf(v,leaf,f'{path}[{i}]'))
        return found
    for label,leaf,expected in [
        ('GMLS V2 quality lambda','lambda_v2_quality',0.25),
        ('GMLS V2 route lambda','lambda_v2_route',0.1),
        ('ROI guide lambda','lambda_guide',0.2),
        ('ROI center lambda','lambda_center',2.0),
    ]:
        matches=find_leaf(cfg,leaf)
        checks[leaf]=check(f'{label} = {expected} (exact key)',
            bool(matches) and any(isinstance(v,(int,float)) and abs(v-expected)<1e-8 for _,v in matches))
        if matches: print('  Matches:',matches)
else:
    print('[FAIL] Cannot inspect deployed dual-head configuration')

print('\n=== C. Install manifest and source YAML precedence ===')
manifest = read('Install manifest')
if manifest:
    try:
        m = json.loads(manifest)
        src = m.get('source_yaml', '<UNKNOWN>')
        print('Installed from source YAML:', src)
        checks['source_v2'] = check('Installed from explicitly named V2 YAML',
            str(src).endswith('_scratch100_v2.yaml'))
    except Exception as e:
        print('[FAIL] Manifest invalid:', e)
else:
    print('[WARN] No dual-head install manifest')
order = [
    base / 'uavcb_gmls_v2_hybrid_geo_center_scratch100.yaml',
    base / 'uavcb_gmls_v2_hybrid_geo_center_scratch100_ddp56.yaml',
    base / 'uavcb_gmls_v2_hybrid_geo_center_scratch100_v2.yaml',
]
first = next((a for a in order if (root / a).is_file()), None)
print('First YAML selected by original installer:', first)
if first and not str(first).endswith('_scratch100_v2.yaml'):
    print('[WARN] Installer prefers a non-V2 YAML if reinstalled as originally written.')

print('\n=== OVERALL ===')
needed = ('v2_hook','v2_loss','v2_ramp','v2_formula','v2_runner',
          'dual_inherit','dual_indep','dual_hook','dual_runner',
          'dual_val','cfg_v2','cfg_top2','cfg_global','cfg_dual',
          'lambda_v2_quality','lambda_v2_route','lambda_guide','lambda_center')
failed = [k for k in needed if not checks.get(k)]
if failed:
    print('[NOT VERIFIED] Missing / failed checks:', ', '.join(failed))
    print('Do NOT assume full GeoCenter V2 is active in dual-head training.')
    sys.exit(1)
print('[STATIC CODE PASS] V2 + dual-head critical code/config markers are present.')
print('This is not GPU forward/backward or live checkpoint verification.')
