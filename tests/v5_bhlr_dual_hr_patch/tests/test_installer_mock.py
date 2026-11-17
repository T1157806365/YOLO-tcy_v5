from pathlib import Path
import tempfile
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
anchor='''    # HVAD_GEOHR_V1_FACTORY_INTEGRATION
    if bool((model_cfg.get("hvad", {}) or {}).get("enabled", False)):
        from hvad.runtime import attach_hvad
        model = attach_hvad(model, model_cfg["hvad"], verbose=verbose)

    return model
'''
with tempfile.TemporaryDirectory() as root:
    p=Path(root)
    (p/'models/modules').mkdir(parents=True)
    (p/'hvad').mkdir()
    (p/'models/gtga_bhlr_factory.py').write_text('def build_model_from_config(model_cfg, verbose):\n'+anchor)
    (p/'models/modules/bhlr_v2.py').write_text('class MultiScaleBHLRv2: pass\n')
    (p/'hvad/runtime.py').write_text('def attach_hvad(model,cfg,verbose):return model\n')
    cfg={"model":{"name":"yolo26m-p2","pretrained":True,"bhlr":{"enabled":True,"detail_channels":48,"stem_channels":16,"high_to_semantic_ratio":3},"hvad":{"enabled":True}},"train":{"device":"5,6","epochs":100,"stage2":{"enabled":False}},"output":{"project":"runs/test","name":"test"}}
    x=p/'configs/experiments/hvad_v1/uavcb_yolo26m_p2_gtga_bhlr_tri_gate_hvad_100e.yaml'
    x.parent.mkdir(parents=True)
    x.write_text(yaml.safe_dump(cfg))
    for j in range(2):
        subprocess.run([sys.executable,str(ROOT/'install_bhlr_dual_hr.py'),'--target',str(p)],check=True,capture_output=True)
    fac=(p/'models/gtga_bhlr_factory.py').read_text()
    assert fac.count('# BHLR_DUAL_HR_V1_FACTORY_INTEGRATION')==1
    assert (p/'models/modules/bhlr_dual_hr.py').is_file()
    for mode in ("control","clean","dual","guided"):
        fp=p/f'configs/experiments/bhlr_dual_hr/uavcb_hvad_{mode}_officialPT_100e.yaml'
        yy=yaml.safe_load(fp.read_text())
        assert yy["train"]["batch"]==24
        assert yy["train"]["resume"] is None
        assert yy["train"]["epochs"]==100
        assert yy["model"].get("bhlr_dual_hr",{}).get("mode")== (None if mode=="control" else mode)
    print('[PASS] install to isolated mock repository, idempotence, four YAML')
