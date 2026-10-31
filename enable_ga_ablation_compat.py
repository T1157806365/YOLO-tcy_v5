#!/usr/bin/env python3
"""Minimal compatibility patch for the 100e dual-teacher noAlign and noBHLR YAMLs.

Run ONCE in the YOLO-tcy_v5 repository root after the 100e and dual-teacher
initialization patches. Does not modify YAML, model weights, or training runs.
Saves timestamped backups before writing any source file.
"""
from __future__ import annotations
import ast
from datetime import datetime
from pathlib import Path
import shutil

ROOT=Path.cwd()
PATHS={
    'factory':ROOT/'ga_monoalign_pcgrad/factory.py',
    'metrics':ROOT/'ga_monoalign_pcgrad/metrics.py',
    'preflight':ROOT/'ga_monoalign_pcgrad/preflight.py',
}

ALIGN_OLD='''    u=core.alignment.unit
    if not (u.coarse_gmls_enabled and u.gmls_v2_correction_enabled and u.fine_translation_enabled):
        raise AssertionError('Original GMLS-v2 implementation is not active')'''
ALIGN_NEW='''    # ABLATION_COMPAT_V1: alignment unit is intentionally absent for noAlign.
    if core.alignment_enabled:
        u=core.alignment.unit
        if not (u.coarse_gmls_enabled and u.gmls_v2_correction_enabled and u.fine_translation_enabled):
            raise AssertionError('Original GMLS-v2 implementation is not active')
    elif core.alignment is not None:
        raise AssertionError('noAlign unexpectedly instantiated an alignment unit')'''

METRIC_EXTRA='''    # ABLATION_COMPAT_V1: noAlign measures unwarped GT box overlap; it is NOT learned AlignIoU.
    if not bool(getattr(student.core, 'alignment_enabled', True)):
        values=[]; scanned=0; zero=0
        for batch in loader:
            bsz=int(batch['rgb_semantic_img'].shape[0])
            for i in range(bsz):
                rb=batch['rgb_semantic_bboxes'][batch['rgb_batch_idx']==i]
                tb=batch['tir_bboxes'][batch['tir_batch_idx']==i]
                if rb.shape[0]!=1 or tb.shape[0]!=1:
                    continue
                rbox=xywh2xyxy(rb.float())[0]
                tbox=xywh2xyxy(tb.float())
                overlap=float(box_iou_single(rbox,tbox).max().item())
                values.append(overlap)
                zero+=int(overlap<=1e-12)
            scanned+=bsz
            if scanned>=max_images:
                break
        if not values:
            raise RuntimeError('No valid RGB/TIR paired GT for noAlign baseline')
        value=sum(values)/len(values)
        return {'align_iou':value, 'unaligned_identity_iou':value,
                'align_zero':zero,'align_pairs':len(values),
                'align_images_scanned':scanned,'align_gate_images':0}
'''

PREFLIGHT_OLD='''    for key in ('coarse_offset_gmls','coarse_offset_dmls','coarse_offset_gmls_v2',
                'gmls_v2_choose_dmls','sampling_scale','sampling_translation'):
        x=original['alignment_info'][key];y=wrapped['alignment_info'][key]
        if x is None or y is None:raise AssertionError(f'GMLS-v2 debug missing {key}')
        if not torch.equal(x,y):raise AssertionError(f'Geometry changed {key}')'''
PREFLIGHT_NEW='''    if net.core.alignment_enabled:
        for key in ('coarse_offset_gmls','coarse_offset_dmls','coarse_offset_gmls_v2',
                    'gmls_v2_choose_dmls','sampling_scale','sampling_translation'):
            x=original['alignment_info'][key];y=wrapped['alignment_info'][key]
            if x is None or y is None:raise AssertionError(f'GMLS-v2 debug missing {key}')
            if not torch.equal(x,y):raise AssertionError(f'Geometry changed {key}')
    elif original['alignment_info'].get('enabled') is not False:
        raise AssertionError('noAlign unexpectedly returned alignment geometry')'''


def once(src,old,new,label):
    if new in src:return src
    if src.count(old)!=1:
        raise RuntimeError(f'{label}: expected exact single match, found {src.count(old)}; no files changed')
    return src.replace(old,new,1)


def main():
    for p in PATHS.values():
        if not p.is_file():raise FileNotFoundError(p)
    old={k:p.read_text(encoding='utf-8') for k,p in PATHS.items()}
    new=dict(old)

    # Original code forbids BHLR=false. Preserve required scale definitions,
    # but permit disabling the actual high-resolution module for an ablation.
    new['factory']=once(new['factory'],
        "    if a.get('apply_scales')!=[3,4,5] or b.get('scales')!=[3] or not b.get('enabled'):\n"
        "        raise ValueError('Keep original P3/P4/P5 alignment and BHLR-P3')",
        "    if a.get('apply_scales')!=[3,4,5] or b.get('scales')!=[3]:\n"
        "        raise ValueError('Keep the original P3/P4/P5 alignment and BHLR-P3 scale specifications')",
        'factory enable noBHLR')
    if 'NOALIGN_ABLATION_V1' not in new['factory'] and 'ABLATION_COMPAT_V1' not in new['factory']:
        new['factory']=once(new['factory'],ALIGN_OLD,ALIGN_NEW,'factory enable noAlign')

    if 'NOALIGN_ABLATION_V1' not in new['metrics'] and 'ABLATION_COMPAT_V1' not in new['metrics']:
        # restrict substitution to geometry_iou by splitting at its definition
        marker='def geometry_iou(student,loader,device,amp=False,max_images=96):\n'
        if new['metrics'].count(marker)!=1:raise RuntimeError('metrics.py geometry_iou definition changed')
        pre,post=new['metrics'].split(marker,1)
        post=once(post,'    student.eval()\n','    student.eval()\n'+METRIC_EXTRA,'metrics noAlign')
        new['metrics']=pre+marker+post
    if 'NOALIGN_ABLATION_V1' not in new['preflight'] and 'ABLATION_COMPAT_V1' not in new['preflight']:
        new['preflight']=once(new['preflight'],PREFLIGHT_OLD,PREFLIGHT_NEW,'preflight noAlign geometry')
    old_gradient="    if not has_grad(net.core.alignment.unit):raise AssertionError('GMLS-v2 not updated')"
    new_gradient="    if net.core.alignment_enabled and not has_grad(net.core.alignment.unit):raise AssertionError('GMLS-v2 not updated')"
    new['preflight']=once(new['preflight'],old_gradient,new_gradient,'preflight noAlign gradients')

    # No code is written until all AST checks pass.
    for k,src in new.items():ast.parse(src,filename=str(PATHS[k]))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    for k,p in PATHS.items():
        if new[k]==old[k]:
            print('[UNCHANGED]',p)
            continue
        bk=p.with_name(p.name+'.bak_ablations_'+stamp)
        shutil.copy2(p,bk)
        p.write_text(new[k],encoding='utf-8')
        print('[BACKUP]',bk)
        print('[PATCH]',p)
    print('[PASS] Both ablations supported; no source training YAML or checkpoint changed')
    print('[NOTE] noAlign align_iou logs report unwarped GT overlap, NOT learned alignment')

if __name__=='__main__': main()
