# YOLO-tcy_v5 蒸馏训练 GPU / Batch 自由选择补丁

该补丁只改 `distill_rgbt/train.py` 末尾 CLI 入口，不改 `validate_cfg()`、训练循环、DDP、学生网络、教师权重、蒸馏损失、模型工厂或验证脚本。保留已有 AG-LHFE + KD 集成。

## 安装（仅一次）

将 ZIP 解压到项目根目录后：

```bash
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python install_flexible_distill_cli.py --check
python install_flexible_distill_cli.py --apply
```

备份在 `distill_rgbt/train.py.before_flexible_cli_v1`；需要恢复时用 `python install_flexible_distill_cli.py --restore`。安装前检查保持原样，不动六组 YAML。

## 直接运行，仍使用原来的 python -m distill_rgbt.train

**单卡 GPU2 / 总 BS16 / 自动累积8 / 有效 Batch128**：

```bash
CFG=configs/experiments/ag_lhfe_kd/KD_B_align_100e.yaml
python -m distill_rgbt.train --cfg "$CFG" --device 2 --batch 16 --check
python -m distill_rgbt.train --cfg "$CFG" --device 2 --batch 16 --smoke-test
python -u -m distill_rgbt.train --cfg "$CFG" --device 2 --batch 16
```

**双卡 GPU5、6 / 总 BS16 / 每卡 BS8 / 自动累积8**：

```bash
python -u -m distill_rgbt.train \
  --cfg configs/experiments/ag_lhfe_kd/KD_A_baseline_100e.yaml \
  --device 5,6 --batch 16
```

**双卡每卡 BS16 / 总 BS32 / 累积4**：

```bash
python -u -m distill_rgbt.train \
  --cfg configs/experiments/ag_lhfe_kd/KD_D_align_bhlr_100e.yaml \
  --device 5,6 --batch-per-gpu 16
```

也可运行 `bash train_flexible_distill.sh KD_B_align_100e 2 16`，自动 check、smoke-test、nohup、PID、日志。

## 选项

- `--device 2` 或 `--device 5,6` / `--device 0,1,2`：物理 GPU ID；多卡自动 DDP，透传参数给所有子进程。
- `--batch N`：所有卡合计的 **global microbatch**。总 BS 必须能被卡数整除。
- `--batch-per-gpu N`：每卡 N，自动换算总 BS；与 `--batch` 互斥。
- `--effective-batch 128`：累积目标；默认继承 YAML `expected_effective_batch`（当前均为128）。
- `--accumulate N`：手动指定梯度累积，覆盖自动计算。
- `--strict-effective-batch`：当目标有效 BS 无法被总 BS 整除时拒绝启动。否则取最接近的整数累积并显式警告实际有效 BS。
- `--check` / `--smoke-test`：原有用法不变。

示例：GPU `[0,1,2]`、全局 Batch18，128/18 无法整除，会选择累积7、实际有效 Batch126，并打印警告。使用 `--strict-effective-batch` 则直接报错，避免不同消融的 batch 意外漂移。

## 注意

- 仅通过静态、语法、纯计算单测和与已有 AG-LHFE-KD 补丁的安装兼容性测试；**未在用户 CUDA 服务器上实际训练**。
- 不改旧 YAML；每次运行将最终的设备/BS/累积配置写进训练 checkpoint 的 `config`，便于追溯。
- 不自动改运行输出目录。如果该实验的输出目录已经有数据，原训练器仍拒绝覆盖，以保护既有模型权重。要重新训练，请改 `output.name` 或用原来的 resume 机制。
- 目标为相同有效 Batch 时，不同 GPU 数量会改变各卡实际微批量和 BN 统计；严格消融仍建议统一 GPU 数、总/单卡 BS、累积策略。
