# YOLO-tcy_v5/main — 四组消融 + 旧版完整蒸馏训练功能（DDP 修复版）

**针对错误** `ValueError: This matched four-arm experiment is single GPU only`。旧四组训练入口强制单卡，且没有旧版 Mono/PCGrad 等功能。本次替换 `distill_rgbt/` 独立训练包，**不改原始 `models/` / `engine/` / `datasets/`**。

依赖：GitHub `T1157806365/YOLO-tcy_v5` `main`，Python/Ultralytics/Torch 等沿用用户已有 `tcy` 环境；UAV-CB 配对图像和 RGB/TIR 教师的本地权重必须存在。

## 四组消融（蒸馏协议相同）

| 配置 | GMLS/DMLS | BHLR | KD | Mono Loss | PCGrad |
|---|---|---|---|---|---|
| `A_baseline_KD100e.yaml` | OFF | OFF | ON | ON | **自动不执行**（无对齐损失）|
| `B_align_only_KD100e.yaml` | ON | OFF | ON | ON | ON |
| `C_hr_only_KD100e.yaml` | OFF | ON（自引导） | ON | ON | **自动不执行**（无对齐损失）|
| `D_align_hr_KD100e.yaml` | ON | ON（几何引导） | ON | ON | ON |

- 所有组都从原始 YOLO 官方预训练起点 + **RGB/TIR 双教师 Backbone 初始化**，没有加载已有 RGB-T 完整模型 best.pt。
- `distill.teacher_init_scope: backbone_only`，可修改为 `full`（与旧版 FullInit 类似，复制 RGB full 主干+Neck+Detect）；为消融公平，四组必须统一。
- KD：RGB/TIR 自己原生坐标的 GT ROI 上，教师可信度 (`min_conf`, `min_gt_iou`) 筛选，P3 空间注意力 KL。训练教师冻结且懒加载。
- Mono Loss：两套训练期辅助 Neck/Detect，按两种模态独立检测监督；保持原版融合推理路径。
- PCGrad：旧版 **特征边界梯度投影**，只修正与对齐监督冲突的 Mono+KD 辅助梯度；不修改检测损失梯度。无对齐时自然失效。这意味着这四组的对比体现了“模块 + 其可用辅助优化”差异；如需要纯结构消融，应将所有 YAML 的 `pcgrad.enabled: false` 后另做严格对照。
- 100 Epoch，SGD 0.005，MultiStep (50/80)，单阶段。**训练器支持**将所有 YAML 统一改为 `train.epochs: 300`, `train.stage2.enabled: true`, `train.stage2.epochs: 200`，以沿用原版两阶段；此时应同时改动 LR 调度的 `lr_milestones` 和 KD 开关时序。100e 结果不能和原版 300+200e 直接比较。
- **DDP：默认物理 GPU 5,6，`train.batch: 16` 为全局 microbatch，两卡各 8；`train.accumulate: 2`，总有效 Batch = 32。** 若用单卡 GPU3，改 `train.device: '3'`，保持 batch=16 和 accumulate=2，有效 Batch 仍为 32。

## 安装到恢复的 `main`

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5

git status --short
git switch -c experiment/distill-ddp-four-arm-v2  # 如已在实验分支则跳过

# 将 zip 放到仓库根目录
unzip -o RGBT_Distill_DDP_Full_v2.zip
```

覆盖的是之前下载的 `distill_rgbt/` 独立实验代码；不会触碰 `models/`、`engine/`、历史权重文件。

## 检查、冒烟测试、训练

先确认两位教师权重的路径（四个 YAML 中 `distill.teachers.rgb/tir`），然后：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
CFG=configs/experiments/distill_rgbt_4way/D_align_hr_KD100e.yaml
python -m distill_rgbt.train --cfg "$CFG" --check
python -m distill_rgbt.train --cfg "$CFG" --smoke-test
```

`--check` 不实际加载模型；`--smoke-test` 用第一张真实配对训练图片做 forward/backward（在第一张配置 GPU 上；不会启动多卡训练），同时检查对齐和 BHLR 梯度。若教师权重/数据不存在会明确报错。

通过后：

```bash
mkdir -p logs pid
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  python -u -m distill_rgbt.train --cfg "$CFG" \
  > logs/D_align_hr_KD100e.log 2>&1 &
echo $! > pid/D_align_hr_KD100e.pid
tail -n 30 -f logs/D_align_hr_KD100e.log
```

训练器自动识别 `device: '5,6'`，打印 `[DDP] Physical GPUs [5, 6] => torchrun 2 processes`，然后使用 `torch.distributed.run` 启动。不要手动重复 `torchrun`。将 `CFG` 依次改为 `A_baseline_KD100e.yaml`、`B_align_only_KD100e.yaml`、`C_hr_only_KD100e.yaml`，即可训练四组。每组输出路径不同。

## 训练功能与日志

- DDP 自动启动、`no_sync()` 梯度累积（包括最后不足 `accumulate` 的 batch）、梯度裁剪、AMP、随机种子、SGD、MultiStep/Cosine 调度。
- 双教师预训练初始化；Mono Loss；冻结教师的 ROI-KD；特征级 PCGrad（支持 AMP 缩放比例）；单阶段/两阶段。
- 每轮 Val Fusion AP50/AP75/AP50:95；对齐开启时默认每轮 96 张子集 AlignIoU；Mono 开启时每 5 轮验证 RGB/TIR 辅助检测分支。
- 打印 `weight_mono`、`weight_kd`、两名教师被筛中比例、PCGrad 冲突率、对齐与 BHLR 梯度范数。
- 输出 `history_stage1.json`、`history_stage1.csv`、`training_summary.json`。
- `best.pt`、`last.pt`、`stage1_best.pt`、`stage1_last.pt`；启用第二阶段后有 `stage2_*`。`model_state_dict` 为 **原始 core.state_dict()**，不包含辅助头，可由 `val_rlsfa_gtga_bhlr_v2.py` 严格加载。额外训练用的辅助头参数独立保存在 `auxiliary_state_dict`。
- 断点续训（**Stage 1**）：将对应 YAML 的 `train.resume` 设为之前同一实验的 `.../weights/stage1_last.pt`；重启会加载优化器、调度器、AMP scaler 和模型/辅助头。建议在单独实验分支测试断点功能。Stage2 从 Stage1 best 启动，当前不支持 Stage2 中途断点续训。

## 本地检查范围与限制

交付前使用 CPU 执行：Python 编译、四组 YAML 互斥开关/相同策略合同检查、ROI-KD 梯度、PCGrad 投影数学检查、主网络兼容检查点格式测试。**未在用户实际 CUDA + UAV-CB 环境跑通一轮，不能声称端到端训练已验证。** 先 `--check` 与 `--smoke-test`，如报错请提供完整 traceback。
