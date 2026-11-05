#!/usr/bin/env python3
"""Re-create matched configs. The generated YAML files are already bundled."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'configs/experiments/ag_lhfe_v1'
OUT.mkdir(parents=True, exist_ok=True)

# Self-contained experiment configuration (matching the original main GMLS/DMLS settings).
if True:
    base = {
      'model': {'variant':'rlsfa_gtga_centered_bhlr_v2','name':'yolo26m','nc':1,'pretrained':True,'fusion':'add','align_mode':'bilinear',
      'rlsfa_gtga_centered':{
        'enabled': True, 'type':'rlsfa_gtga_centered','reference_scale':3,'apply_scales':[3,4,5],
        'hidden_channels':32,'coarse_radius':20,'coarse_temperature':0.10,'use_targetness_for_correlation':True,
        'coarse_local_soft_enabled':False,'coarse_local_radius':6,'coarse_local_temperature':0.10,
        'coarse_gmls_enabled':True,'coarse_gmls_topk':3,'coarse_gmls_nms_radius':4,
        'coarse_gmls_aggregation_temperature':0.12,'coarse_gmls_global_prior':0.60,
        'coarse_gmls_feature_weight':0.65,'coarse_gmls_target_weight':0.35,
        'gmls_v2_correction_enabled':True,'gmls_v2_quality_hidden':32,'gmls_v2_route_logit_margin':0.20,
        'fine_translation_enabled':True,'fine_translation_radius':2,'fine_translation_temperature':0.05,
        'anchor_power':4.,'moment_power':2.,'local_window_sigma':0.1,'moment_border_margin':2,
        'sigma_min':0.005,'scale_min':0.5,'scale_max':3.0,'padding_mode':'border','align_corners':True,
        'reliability_gate':{'enabled':False},
      },
      'bhlr':{'enabled':True,'scales':[3],'high_to_semantic_ratio':3,
              'detail_channels':48,'stem_channels':16,'guide_channels':24,
              'guidance_mode':'alignment','detach_alignment_guidance':True}},
      'data':{'rgb':'/mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5/configs/datasets/UAV-CB-RGB.yaml',
              'tir':'/mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5/configs/datasets/UAV-CB-TIR.yaml',
              'rgb_imgsz':1920,'rgb_semantic_imgsz':640,'tir_imgsz':640,'pair_mode':'relative'},
      'train':{'device':'5,6','batch':8,'batch_mode':'global','workers':4,'epochs':100,
               'optimizer':'SGD','lr0':0.005,'lrf':0.01,'momentum':0.937,'weight_decay':0.0005,
               'amp':True,'seed':0,'deterministic':True,'grad_clip':10.0,
               'patience':100,'early_stop_min_delta':1e-6,'resume':None,
               'ddp_find_unused_parameters':True,
               'stage2':{'enabled':False,'epochs':200,'lr0':0.001,'lrf':0.01,
                         'patience':50,'early_stop_min_delta':1e-6}},
      'augment':{'fliplr':0.5,'flipud':0.0},
      'loss':{'box':7.5,'cls':0.5,'dfl':1.5,
              'alignment':{'enabled':True,'lambda_alignment':0.2,'lambda_anchor':1.,'lambda_coarse':1.,
                           'lambda_scale':1.,'lambda_targetness':0.2,'smooth_l1_beta':1.,
                           'targetness_alpha':0.75,'targetness_gamma':2.,
                           'gtga_gaussian_sigma_factor':0.25,'gtga_gaussian_sigma_min_px':1.,
                           'clamp_gt_translation':True,'lambda_v2_quality':0.25,'lambda_v2_route':0.1,
                           'v2_quality_iou_weight':0.8,'v2_quality_advantage':0.05,
                           'v2_route_target_temperature':0.05}},
      'output':{'project':'runs/ag_lhfe_v1','name':'TEMP'},
    }
for tag, mode, edges in [
    ('E0_OriginalBHLR_100e',None,True),
    ('E1_LocalRGB_100e','rgb',True),
    ('E2_LocalAligned_100e','rgb_align',True),
    ('E3_LocalAligned_NoEdges_100e','rgb_align',False),
]:
    cfg=deepcopy(base)
    cfg['output']['name']=f'yolo26m_UAVCB_{tag}_seed0'
    if mode is not None:
        cfg['model']['variant']='rlsfa_gtga_centered_ag_lhfe_v1'
        cfg['model']['bhlr']={'enabled':False,'guidance_mode':'none'}
        cfg['model']['ag_lhfe']={
            'enabled':True,'proposal_mode':mode,'topk':16,'nms_radius':3,
            'patch_size':32,'crop_sizes_high':[48,96],'latent_dim':64,
            'use_edges':edges,'strength_init':0.05,'strength_max':0.20,
            'max_offset':2.5,'splat_radius':2,'positive_radius':2.5,
            'lambda_obj':0.05,'lambda_loc':0.02,
        }
    target=OUT/(tag+'.yaml')
    target.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print(target)



# Original main's complete 300+200-epoch schedule, for final matched runs.
for tag, mode, edges in [
    ('E0_OriginalBHLR_300p200', None, True),
    ('E1_LocalRGB_300p200', 'rgb', True),
    ('E2_LocalAligned_300p200', 'rgb_align', True),
]:
    cfg = deepcopy(base)
    cfg['train']['epochs'] = 300
    cfg['train']['lr0'] = 0.01
    cfg['train']['patience'] = 50
    cfg['train']['stage2'] = {
        'enabled': True, 'epochs': 200, 'lr0': 0.001,
        'lrf': 0.01, 'patience': 50, 'early_stop_min_delta': 1e-6,
    }
    cfg['output']['name'] = f'yolo26m_UAVCB_{tag}_seed0'
    if mode is not None:
        cfg['model']['variant'] = 'rlsfa_gtga_centered_ag_lhfe_v1'
        cfg['model']['bhlr'] = {'enabled': False, 'guidance_mode': 'none'}
        cfg['model']['ag_lhfe'] = {
            'enabled': True, 'proposal_mode': mode, 'topk': 16, 'nms_radius': 3,
            'patch_size': 32, 'crop_sizes_high': [48,96], 'latent_dim': 64,
            'use_edges': edges, 'strength_init': 0.05, 'strength_max': 0.2,
            'max_offset': 2.5, 'splat_radius': 2, 'positive_radius': 2.5,
            'lambda_obj': 0.05, 'lambda_loc': 0.02,
        }
    path = OUT/(tag+'.yaml')
    path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    print(path)
