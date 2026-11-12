"""Strict validation of new student and two single-modal auxiliary heads."""
from __future__ import annotations

# PHYSICAL_GPU_VAL_BEGIN
# --device N 始终对应 nvidia-smi 中的物理 GPU N
# 必须在 torch 和项目模块导入之前执行

import os as _gpu_os
import sys as _gpu_sys
import subprocess as _gpu_subprocess


def _configure_physical_gpu():
    args = _gpu_sys.argv[1:]
    requested = "0"

    for i, arg in enumerate(args):
        if arg == "--device" and i + 1 < len(args):
            requested = args[i + 1]
            break
        if arg.startswith("--device="):
            requested = arg.split("=", 1)[1]
            break

    if requested.lower() == "cpu":
        _gpu_os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    if not requested.isdigit():
        raise ValueError(
            f"--device 必须是物理 GPU 编号或 cpu，收到: {requested}"
        )

    physical_id = int(requested)

    output = _gpu_subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    )

    gpu_map = {}
    for line in output.strip().splitlines():
        index, uuid = [x.strip() for x in line.split(",", 1)]
        gpu_map[int(index)] = uuid

    if physical_id not in gpu_map:
        raise RuntimeError(
            f"物理 GPU {physical_id} 不存在，可用编号: {sorted(gpu_map)}"
        )

    # 用 UUID 锁定物理 GPU，避免 CUDA 与 nvidia-smi 顺序不一致
    _gpu_os.environ["CUDA_VISIBLE_DEVICES"] = gpu_map[physical_id]

    print(
        f"[PHYSICAL GPU] {physical_id} -> "
        f"{gpu_map[physical_id]} -> cuda:0",
        flush=True,
    )

    return "0"


_PHYSICAL_VAL_DEVICE = (
    _configure_physical_gpu()
    if __name__ == "__main__"
    else "0"
)
# PHYSICAL_GPU_VAL_END

import argparse,json
import torch
from pathlib import Path
from datasets.dual_resolution_rgbt_dataset import build_dual_resolution_dataset,build_dual_resolution_dataloader
from engine.runtime import select_device
from engine.evaluator import validate_metrics
from .factory import load_checkpoint
from .metrics import mono_metrics,geometry_iou


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--device',default='0');p.add_argument('--batch',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conf',type=float,default=.001);p.add_argument('--iou',type=float,default=.70)
    p.add_argument('--save-dir',default=None)
    p.add_argument('--geometry-all',action='store_true',help='Calculate alignment IoU on entire split')
    p.add_argument(
        "--bhlr-injection-position",
        default="checkpoint",
        choices=("checkpoint", "pre_fusion", "post_fusion"),
        help="For same-weight counterfactual tests only; does not change checkpoint",
    )
    p.add_argument(
        "--disable-bhlr-residual",
        action="store_true",
        help="Zero BHLR gamma in-memory, leaving all other weights intact",
    )
    args=p.parse_args()
    device=select_device(_PHYSICAL_VAL_DEVICE)
    model,ck,cfg=load_checkpoint(args.weights,device)
    if args.bhlr_injection_position != "checkpoint":
        model.core.bhlr_injection_position = args.bhlr_injection_position
    if args.disable_bhlr_residual:
        if model.core.bhlr is None:
            raise RuntimeError("Checkpoint has no BHLR module to disable")
        with torch.no_grad():
            for adapter in model.core.bhlr.adapters.values():
                adapter.gamma_raw.zero_()
    print("[ABLATION] BHLR injection:", model.core.bhlr_injection_position,
          "residual_off:", args.disable_bhlr_residual, flush=True)
    d=cfg['data']
    data=build_dual_resolution_dataset(rgb_yaml=d['rgb'],tir_yaml=d['tir'],split=args.split,
        rgb_high_imgsz=int(d['rgb_imgsz']),rgb_semantic_imgsz=int(d['rgb_semantic_imgsz']),
        tir_imgsz=int(d['tir_imgsz']),pair_mode=d.get('pair_mode','relative'),
        augment=False,tir_channels=3,strict_pair=True)
    loader=build_dual_resolution_dataloader(data,batch_size=args.batch,workers=args.workers,
        shuffle=False,pin_memory=device.type=='cuda')
    main=validate_metrics(model,loader,device,amp=device.type=='cuda',
        conf_thres=args.conf,iou_thres=args.iou,max_det=300,preview_num=0)
    mono=mono_metrics(model,loader,device,amp=device.type=='cuda')
    geom=geometry_iou(model,loader,device,max_images=len(data) if args.geometry_all else 96)
    res={'split':args.split,'checkpoint':str(Path(args.weights).resolve()),
         'bhlr_injection_position':model.core.bhlr_injection_position,
         'bhlr_residual_disabled':bool(args.disable_bhlr_residual),
         'student_scratch':True,'gmlsv2':True,**main,**mono,**geom}
    path=Path(args.save_dir) if args.save_dir else Path(args.weights).resolve().parent.parent/f'val_{args.split}_monoalign'
    if path.exists() and (path/'metrics.json').exists():
        raise FileExistsError(f'Refusing to overwrite metrics: {path}/metrics.json')
    path.mkdir(parents=True,exist_ok=True)
    (path/'metrics.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(res,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
