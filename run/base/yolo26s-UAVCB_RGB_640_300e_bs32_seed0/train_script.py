# ============================================================
# YOLO26n RGB - Anti-UAV 正式训练脚本
# ============================================================

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


# ============================================================
# 1. 固定路径
# ============================================================
"""
    conda activate tcy
    cd /mnt/sda/taochangyong/Projects/YOLO26-AntiUAV
    /mnt/sda/taochangyong/.conda/envs/tcy/bin/python \
    train_yolo26_rgb.py \
    --gpu 4 \
    --batch 32 \
    --epochs 300 \
    --imgsz 640\
    --workers 4 \
    --seed 0 \
    > train_yolo26s_UAV-CB-RGB_1920_300e_bs32_seed0_new.log 2>&1 &
    tail -f train_yolo26s_UAV-CB-RGB_1920_300e_bs32_seed0_new.log
"""
MODEL = "yolo26s.pt"

DATA_YAML = Path(
    "/mnt/sda/taochangyong/Projects/"
    "YOLO26-AntiUAV/UAV-CB-RGB.yaml"
)

DATA_ROOT = Path(
    "/mnt/sda/taochangyong/Projects/"
    "datasets/UAV-CB"
)

PROJECT = Path(
    "/mnt/sda/taochangyong/Projects/"
    "YOLO26-AntiUAV/runs"
)

BASE_RUN_NAME = "train_yolo26s_UAV-CB-UAV-RGB"


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="YOLO26s RGB Anti-UAV training"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="5",
        help="GPU编号，例如 2,3,5"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=48,
        help="Global batch size"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="训练和测试输入尺寸，例如 640、1080、1280、1920"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="训练轮数"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子"
    )

    return parser.parse_args()


# ============================================================
# 3. 自动生成实验名称
# ============================================================

def get_next_run_name(project, base_name):

    run_name = base_name
    run_dir = project / run_name

    if not run_dir.exists():
        return run_name, run_dir

    index = 1

    while True:

        run_name = f"{base_name}_{index}"
        run_dir = project / run_name

        if not run_dir.exists():
            return run_name, run_dir

        index += 1


# ============================================================
# 4. 保存文本
# ============================================================

def save_text(path, text):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(str(text))


# ============================================================
# 5. JSON 安全转换
# ============================================================

def json_safe(obj):

    if obj is None:
        return None

    if isinstance(
        obj,
        (str, int, float, bool)
    ):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):

        return {
            str(k): json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple)
    ):

        return [
            json_safe(v)
            for v in obj
        ]

    if hasattr(obj, "item"):

        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist"):

        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


# ============================================================
# 6. 保存 JSON
# ============================================================

def save_json(path, data):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_safe(data),
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# 7. 执行 Shell 命令
# ============================================================

def run_command(command):

    try:

        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True
        )

        output = result.stdout

        if result.stderr:
            output += "\n" + result.stderr

        return output

    except Exception as e:

        return str(e)


# ============================================================
# 8. SHA256
# ============================================================

def sha256_file(path):

    path = Path(path)

    if not path.exists():
        return None

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 9. 数据集统计
# ============================================================

def dataset_statistics(split):

    folder = (
        DATA_ROOT
        / "visible"
        / split
    )

    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    if not folder.exists():

        return {
            "split": split,
            "folder": str(folder),
            "error": "directory does not exist"
        }

    images = sorted([
        p
        for p in folder.iterdir()
        if p.suffix.lower()
        in image_exts
    ])

    image_count = len(images)

    label_count = 0

    missing_label_count = 0

    empty_label_count = 0

    total_objects = 0

    invalid_lines = []

    class_counts = {}

    for image_path in images:

        label_path = image_path.with_suffix(
            ".txt"
        )

        if not label_path.exists():

            missing_label_count += 1
            continue

        label_count += 1

        try:

            text = label_path.read_text(
                encoding="utf-8"
            ).strip()

        except Exception as e:

            invalid_lines.append({

                "file":
                    str(label_path),

                "error":
                    str(e)
            })

            continue

        # 空 TXT = 背景图
        if not text:

            empty_label_count += 1
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            1
        ):

            parts = line.strip().split()

            if len(parts) != 5:

                invalid_lines.append({

                    "file":
                        str(label_path),

                    "line":
                        line_number,

                    "content":
                        line,

                    "error":
                        "label does not contain 5 values"
                })

                continue

            try:

                class_id = int(
                    float(parts[0])
                )

                x, y, w, h = map(
                    float,
                    parts[1:]
                )

                if class_id < 0:

                    raise ValueError(
                        "class id < 0"
                    )

                if not (
                    0 <= x <= 1
                    and
                    0 <= y <= 1
                    and
                    0 <= w <= 1
                    and
                    0 <= h <= 1
                ):

                    raise ValueError(
                        "bbox outside [0, 1]"
                    )

                total_objects += 1

                class_counts[class_id] = (
                    class_counts.get(
                        class_id,
                        0
                    )
                    + 1
                )

            except Exception as e:

                invalid_lines.append({

                    "file":
                        str(label_path),

                    "line":
                        line_number,

                    "content":
                        line,

                    "error":
                        str(e)
                })

    return {

        "split":
            split,

        "folder":
            str(folder),

        "images":
            image_count,

        "label_files":
            label_count,

        "missing_labels":
            missing_label_count,

        "background_images":
            empty_label_count,

        "objects":
            total_objects,

        "class_counts":
            class_counts,

        "invalid_label_lines":
            len(invalid_lines),

        "invalid_examples":
            invalid_lines[:100]
    }


# ============================================================
# 10. 保存环境信息
# ============================================================

def save_environment(
    run_dir,
    gpu_string,
    train_args
):

    gpu_data = []

    if torch.cuda.is_available():

        for i in range(
            torch.cuda.device_count()
        ):

            try:

                prop = (
                    torch.cuda
                    .get_device_properties(i)
                )

                gpu_data.append({

                    "gpu_id":
                        i,

                    "name":
                        prop.name,

                    "memory_GB":
                        round(
                            prop.total_memory
                            / 1024**3,
                            2
                        )
                })

            except Exception as e:

                gpu_data.append({

                    "gpu_id":
                        i,

                    "error":
                        str(e)
                })

    environment = {

        "time":
            datetime.now().isoformat(),

        "python_version":
            sys.version,

        "python_executable":
            sys.executable,

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "ultralytics_version":
            ultralytics.__version__,

        "cuda_available":
            torch.cuda.is_available(),

        "torch_cuda_version":
            torch.version.cuda,

        "cudnn_version":
            torch.backends.cudnn.version(),

        "cuda_device_count":
            torch.cuda.device_count(),

        "selected_gpu":
            gpu_string,

        "working_directory":
            os.getcwd(),

        "command":
            " ".join(sys.argv),

        "training_arguments":
            train_args,

        "available_gpus":
            gpu_data
    }

    save_json(
        run_dir
        / "experiment_environment.json",
        environment
    )

    # GPU 当前状态
    save_text(
        run_dir
        / "nvidia_smi_before_training.txt",

        run_command(
            "nvidia-smi"
        )
    )

    # 操作系统、CPU、内存
    save_text(
        run_dir
        / "system_info.txt",

        run_command(
            "uname -a; "
            "echo; "
            "lscpu; "
            "echo; "
            "free -h"
        )
    )

    # Python 包
    save_text(
        run_dir
        / "pip_freeze.txt",

        run_command(
            f"{sys.executable} "
            "-m pip freeze"
        )
    )

    # Ultralytics 信息
    save_text(
        run_dir
        / "ultralytics_package.txt",

        run_command(
            f"{sys.executable} "
            "-m pip show ultralytics"
        )
    )

    # Torch 环境
    save_text(
        run_dir
        / "torch_environment.txt",

        run_command(
            f"{sys.executable} "
            "-m torch.utils.collect_env"
        )
    )


# ============================================================
# 11. 保存 Test 指标 CSV
# ============================================================

def save_test_csv(
    path,
    summary
):

    path = Path(path)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Precision",
            "Recall",
            "mAP50",
            "mAP75",
            "mAP50-95"
        ])

        writer.writerow([
            summary.get("Precision"),
            summary.get("Recall"),
            summary.get("mAP50"),
            summary.get("mAP75"),
            summary.get("mAP50_95")
        ])


# ============================================================
# 12. 主程序
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    # 防止误输入中文逗号
    args.gpu = (
        args.gpu
        .replace("，", ",")
        .replace(" ", "")
    )

    # ========================================================
    # 实验名称
    # ========================================================

    base_run_name = (
        f"yolo11m-RGB_"
        f"{args.imgsz}_"
        f"{args.epochs}e_"
        f"bs{args.batch}_"
        f"seed{args.seed}"
    )

    PROJECT.mkdir(
        parents=True,
        exist_ok=True
    )

    RUN_NAME, RUN_DIR = (
        get_next_run_name(
            PROJECT,
            base_run_name
        )
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=False
    )


    # ========================================================
    # 训练参数
    # ========================================================

    TRAIN_ARGS = {

    "data": str(DATA_YAML),

    "imgsz": args.imgsz,
    "rect": False,

    "epochs": args.epochs,

    "batch": args.batch,

    "device": args.gpu,

    "workers": args.workers,

    "cache": False,

    # Optimizer
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,

    # Augmentation
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "close_mosaic": 10,

    # Training
    "pretrained": True,
    "amp": True,
    "seed": args.seed,
    "deterministic": True,

    # =========================
    # Early Stopping
    # =========================
    "patience": 100,

    "val": True,

    # Save
    "save": True,
    "save_period": 10,
    "project": str(PROJECT),
    "name": RUN_NAME,
    "exist_ok": True,
    "plots": True,
}


    # ========================================================
    # 启动信息
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "YOLO11m RGB Anti-UAV 正式训练"
    )

    print("=" * 80)

    print(
        f"模型        : {MODEL}"
    )

    print(
        f"GPU         : {args.gpu}"
    )

    print(
        f"Input       : {args.imgsz}"
    )

    print(
        f"Epochs      : {args.epochs}"
    )

    print(
        f"Global Batch: {args.batch}"
    )

    print(
        f"Workers     : {args.workers}"
    )

    print(
        f"Seed        : {args.seed}"
    )

    print(
        f"实验名称    : {RUN_NAME}"
    )

    print(
        f"保存目录    : {RUN_DIR}"
    )

    print("=" * 80)


    # ========================================================
    # 数据 YAML 检查
    # ========================================================

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"找不到数据集 YAML：\n"
            f"{DATA_YAML}"
        )


    # ========================================================
    # A. 保存训练脚本
    # ========================================================

    shutil.copy2(
        Path(__file__).resolve(),
        RUN_DIR / "train_script.py"
    )


    # ========================================================
    # B. 保存数据 YAML
    # ========================================================

    shutil.copy2(
        DATA_YAML,
        RUN_DIR / "dataset.yaml"
    )


    # ========================================================
    # C. 保存训练配置
    # ========================================================

    save_json(
        RUN_DIR
        / "training_config.json",

        {
            "model":
                MODEL,

            "experiment":
                RUN_NAME,

            "modality":
                "RGB",

            **TRAIN_ARGS
        }
    )


    # ========================================================
    # D. 数据集统计
    # ========================================================

    print(
        "\n正在检查并统计数据集..."
    )

    dataset_stats = {

        "train":
            dataset_statistics(
                "train"
            ),

        "val":
            dataset_statistics(
                "val"
            ),

        "test":
            dataset_statistics(
                "test"
            )
    }

    save_json(
        RUN_DIR
        / "dataset_statistics.json",

        dataset_stats
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "数据集统计"
    )

    print(
        "-" * 80
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        stats = dataset_stats[split]

        if "error" in stats:

            print(
                f"{split}: "
                f"{stats['error']}"
            )

            continue

        print(

            f"{split:5s} | "

            f"Images: "
            f"{stats['images']:6d} | "

            f"Objects: "
            f"{stats['objects']:6d} | "

            f"Background: "
            f"{stats['background_images']:5d} | "

            f"Missing labels: "
            f"{stats['missing_labels']:5d} | "

            f"Invalid: "
            f"{stats['invalid_label_lines']:5d}"
        )

    print(
        "-" * 80
    )


    # ========================================================
    # 如果标签存在严重问题则停止
    # ========================================================

    for split in [
        "train",
        "val",
        "test"
    ]:

        stats = dataset_stats[split]

        if (
            "invalid_label_lines"
            in stats
            and
            stats["invalid_label_lines"] > 0
        ):

            print(
                f"\n警告：{split} "
                f"发现 "
                f"{stats['invalid_label_lines']} "
                f"条异常标签。"
            )


    # ========================================================
    # E. 环境信息
    # ========================================================

    print(
        "\n正在保存运行环境..."
    )

    save_environment(
        RUN_DIR,
        args.gpu,
        TRAIN_ARGS
    )


    # ========================================================
    # F. Hash
    # ========================================================

    save_json(
        RUN_DIR
        / "file_hashes_before_training.json",

        {
            "dataset_yaml":
                sha256_file(
                    DATA_YAML
                ),

            "train_script":
                sha256_file(
                    Path(__file__).resolve()
                )
        }
    )


    # ========================================================
    # G. 加载模型
    # ========================================================

    print(
        "\n正在加载模型..."
    )

    model = YOLO(
        MODEL
    )


    # ========================================================
    # H. 保存模型结构
    # ========================================================

    total_params = sum(
        parameter.numel()
        for parameter
        in model.model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter
        in model.model.parameters()
        if parameter.requires_grad
    )

    save_text(
        RUN_DIR
        / "model_structure.txt",

        str(
            model.model
        )
        + "\n\n"
        + "=" * 80
        + "\n"
        + f"Total parameters: "
        + f"{total_params:,}\n"
        + f"Trainable parameters: "
        + f"{trainable_params:,}\n"
    )

    save_json(
        RUN_DIR
        / "model_parameters.json",

        {
            "model":
                MODEL,

            "total_parameters":
                total_params,

            "trainable_parameters":
                trainable_params
        }
    )


    # ========================================================
    # I. 开始训练
    # ========================================================

    start_time = datetime.now()

    save_text(
        RUN_DIR
        / "training_start_time.txt",

        start_time.isoformat()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "开始训练"
    )

    print(
        "=" * 80
        + "\n"
    )

    try:

        train_results = model.train(
            **TRAIN_ARGS
        )

    except Exception as e:

        failure_time = datetime.now()

        save_text(
            RUN_DIR
            / "TRAINING_FAILED.txt",

            f"""
Training failed

Time:
{failure_time.isoformat()}

Error:
{repr(e)}
"""
        )

        save_text(
            RUN_DIR
            / "nvidia_smi_after_failure.txt",

            run_command(
                "nvidia-smi"
            )
        )

        raise


    # ========================================================
    # J. 训练结束时间
    # ========================================================

    end_time = datetime.now()

    training_seconds = (
        end_time
        - start_time
    ).total_seconds()

    save_json(
        RUN_DIR
        / "training_time.json",

        {
            "start":
                start_time.isoformat(),

            "end":
                end_time.isoformat(),

            "seconds":
                training_seconds,

            "minutes":
                training_seconds / 60,

            "hours":
                training_seconds / 3600
        }
    )

    save_text(
        RUN_DIR
        / "nvidia_smi_after_training.txt",

        run_command(
            "nvidia-smi"
        )
    )


    # ========================================================
    # K. 权重
    # ========================================================

    BEST_PT = (
        RUN_DIR
        / "weights"
        / "best.pt"
    )

    LAST_PT = (
        RUN_DIR
        / "weights"
        / "last.pt"
    )

    if not BEST_PT.exists():

        raise FileNotFoundError(
            f"\n训练结束但没有找到 best.pt：\n"
            f"{BEST_PT}"
        )


    # ========================================================
    # L. 权重 Hash
    # ========================================================

    save_json(
        RUN_DIR
        / "weights_sha256.json",

        {
            "best.pt":
                sha256_file(
                    BEST_PT
                ),

            "last.pt":
                sha256_file(
                    LAST_PT
                )
        }
    )


    # ========================================================
    # M. 测试集评价
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "训练结束，开始使用 best.pt "
        "评价 Test Set"
    )

    print(
        "=" * 80
        + "\n"
    )

    best_model = YOLO(
        str(BEST_PT)
    )

    # 测试固定使用选定GPU中的第一张
    TEST_GPU = (
        args.gpu
        .split(",")[0]
    )

    test_metrics = best_model.val(

        data=str(
            DATA_YAML
        ),

        # 使用 test
        split="test",

        imgsz=args.imgsz,

        rect=False,

        device=TEST_GPU,

        batch=8,

        workers=args.workers,

        # AP计算
        conf=0.001,

        iou=0.7,

        max_det=300,

        # 保存结果
        plots=True,

        save_json=True,

        save_txt=True,

        save_conf=True,

        project=str(
            RUN_DIR
        ),

        name="test",

        exist_ok=True
    )


    # ========================================================
    # N. 提取测试指标
    # ========================================================

    box_metrics = test_metrics.box

    test_summary = {

        "model":
            str(BEST_PT),

        "dataset":
            str(DATA_YAML),

        "split":
            "test",

        "imgsz": args.imgsz,
         "rect": False,

        "device":
            TEST_GPU,

        "Precision":
            getattr(
                box_metrics,
                "mp",
                None
            ),

        "Recall":
            getattr(
                box_metrics,
                "mr",
                None
            ),

        "mAP50":
            getattr(
                box_metrics,
                "map50",
                None
            ),

        "mAP75":
            getattr(
                box_metrics,
                "map75",
                None
            ),

        "mAP50_95":
            getattr(
                box_metrics,
                "map",
                None
            ),

        "speed_ms":
            getattr(
                test_metrics,
                "speed",
                None
            ),

        "results_dict":
            getattr(
                test_metrics,
                "results_dict",
                None
            )
    }


    # ========================================================
    # O. 保存 Test JSON
    # ========================================================

    save_json(
        RUN_DIR
        / "test_summary.json",

        test_summary
    )


    # ========================================================
    # P. 保存 Test CSV
    # ========================================================

    save_test_csv(
        RUN_DIR
        / "test_summary.csv",

        test_summary
    )


    # ========================================================
    # Q. 保存混淆矩阵数值
    # ========================================================

    try:

        confusion_matrix = (
            test_metrics
            .confusion_matrix
            .matrix
        )

        save_json(
            RUN_DIR
            / "test_confusion_matrix.json",

            {
                "matrix":
                    confusion_matrix
            }
        )

    except Exception as e:

        save_text(
            RUN_DIR
            / "confusion_matrix_export_error.txt",

            str(e)
        )


    # ========================================================
    # R. 保存曲线原始数据（如果当前版本提供）
    # ========================================================

    try:

        curve_results = (
            test_metrics
            .curves_results
        )

        save_json(
            RUN_DIR
            / "test_curves_results.json",

            curve_results
        )

    except Exception as e:

        save_text(
            RUN_DIR
            / "curves_export_error.txt",

            str(e)
        )


    # ========================================================
    # S. 最终实验摘要
    # ========================================================

    train_stats = (
        dataset_stats["train"]
    )

    val_stats = (
        dataset_stats["val"]
    )

    test_stats = (
        dataset_stats["test"]
    )

    summary = f"""
================================================================================
YOLO11m RGB Anti-UAV Experiment Summary
================================================================================

Experiment
--------------------------------------------------------------------------------

Run Name:
{RUN_NAME}

Model:
{MODEL}

Task:
Object Detection

Modality:
RGB

Dataset:
Anti-UAV-RGBT

Dataset YAML:
{DATA_YAML}

================================================================================
Training Configuration
================================================================================

GPU:
{args.gpu}

Input Size:
{args.imgsz}

Epochs:
{args.epochs}

Global Batch:
{args.batch}

Workers:
{args.workers}

Optimizer:
SGD

Initial Learning Rate:
0.01

Final LR Factor:
0.01

Momentum:
0.937

Weight Decay:
0.0005

Warmup Epochs:
3.0

Seed:
{args.seed}

AMP:
True

Deterministic:
True

Mosaic:
1.0

Close Mosaic:
10

MixUp:
0.0

================================================================================
Dataset Statistics
================================================================================

Train Images:
{train_stats.get('images')}

Train Objects:
{train_stats.get('objects')}

Train Background Images:
{train_stats.get('background_images')}

Validation Images:
{val_stats.get('images')}

Validation Objects:
{val_stats.get('objects')}

Validation Background Images:
{val_stats.get('background_images')}

Test Images:
{test_stats.get('images')}

Test Objects:
{test_stats.get('objects')}

Test Background Images:
{test_stats.get('background_images')}

================================================================================
Model
================================================================================

Total Parameters:
{total_params:,}

Trainable Parameters:
{trainable_params:,}

================================================================================
Training Time
================================================================================

Start:
{start_time.isoformat()}

End:
{end_time.isoformat()}

Training Seconds:
{training_seconds:.2f}

Training Hours:
{training_seconds / 3600:.4f}

================================================================================
Test Results
================================================================================

Precision:
{test_summary['Precision']}

Recall:
{test_summary['Recall']}

mAP50:
{test_summary['mAP50']}

mAP75:
{test_summary['mAP75']}

mAP50-95:
{test_summary['mAP50_95']}

Inference Speed:
{test_summary['speed_ms']}

================================================================================
Weights
================================================================================

Best:
{BEST_PT}

Last:
{LAST_PT}

================================================================================
"""

    save_text(
        RUN_DIR
        / "EXPERIMENT_SUMMARY.txt",

        summary
    )

    print(summary)


    # ========================================================
    # T. 最终完成标记
    # ========================================================

    save_text(
        RUN_DIR
        / "EXPERIMENT_COMPLETED.txt",

        f"""
Experiment completed successfully.

Run:
{RUN_NAME}

Completed:
{datetime.now().isoformat()}

Best:
{BEST_PT}

Test mAP50:
{test_summary['mAP50']}

Test mAP50-95:
{test_summary['mAP50_95']}
"""
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "全部训练、测试及实验信息保存完成。"
    )

    print(
        f"\n实验目录：\n{RUN_DIR}"
    )

    print(
        "=" * 80
    )