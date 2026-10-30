"""Copy EXACT baseline GMLSv2 YAML, changing only training add-ons/experiment name."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from .factory import validate_cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-cfg',required=True)
    p.add_argument('--rgb-teacher',required=True)
    p.add_argument('--tir-teacher',required=True)
    p.add_argument('--device',default='0,4')
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--accumulate',type=int,default=1)
    p.add_argument('--pretrained',action='store_true',help='Initialize both branches with official YOLO26m pretrained weights')
    p.add_argument('--output',default='configs/experiments/ga_monoalign_pcgrad_v1/uavcb_scratch_gmlsv2.yaml')
    a=p.parse_args()
    base=Path(a.base_cfg)
    data=yaml.safe_load(base.read_text(encoding='utf-8'))
    m=data['model'];m['pretrained']=bool(a.pretrained)
    tr=data['train'];tr.update({'device':a.device,'batch':a.batch,'batch_mode':'global',
        'accumulate':a.accumulate,'init_student_weights':None,'resume':None,
        'ddp_find_unused_parameters':True,'epochs':300})
    tr['stage2'].update({'enabled':True,'epochs':200})
    data['output']={'project':'runs/ga_monoalign_pcgrad_v1',
        'name':('yolo26m_UAVCB_GMLSv2_MonoAlignPCGrad_Pretrained_seed0' if a.pretrained else 'yolo26m_UAVCB_GMLSv2_MonoAlignPCGrad_Scratch_seed0')}
    data['dual_teachers']={'rgb_weights':str(Path(a.rgb_teacher).expanduser().resolve()),
        'tir_weights':str(Path(a.tir_teacher).expanduser().resolve()),
        'min_conf':0.30,'min_gt_iou':0.50,'forward_microbatch':2}
    data['monoalign']={'pcgrad_enabled':True,'alignment_val_images':96,
        'mono_val_every':5, 'minimum_stage1_epochs':130, 'save_joint_gamma':0.10,
        'student_mode':('official_yolo_pretrained' if a.pretrained else 'scratch'),'teacher_kd_type':'target_roi_attention',
        'pcgrad_type':'feature_boundary_projection'}
    # Preserve baseline GMLS-v2 all alignment hyperparameters and losses.
    validate_cfg(data,need_teachers=False)
    dst=Path(a.output)
    if dst.exists():raise FileExistsError(f'Will not overwrite existing config: {dst}')
    out_dir=Path(data['output']['project'])/data['output']['name']
    if out_dir.exists():raise FileExistsError(f'Experiment output already exists: {out_dir}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(yaml.safe_dump(data,allow_unicode=True,sort_keys=False),encoding='utf-8')
    print('[CONFIG]',dst)
    print('[STUDENT]', 'Official YOLO26m pretrained weights for both RGB/TIR branches' if a.pretrained else 'Random init; no pretrained weights')
    print('[TEACHERS]',data['dual_teachers'])
    print('[TRAIN]',tr['epochs'],tr['stage2']['epochs'], 'device',tr['device'],
          'batch',tr['batch'],'accumulate',tr['accumulate'])

if __name__=='__main__':main()
