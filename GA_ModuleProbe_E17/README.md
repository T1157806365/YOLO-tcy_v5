# 第17轮权重：对齐与BHLR模块专训

用途：独立训练 A/B 两组，每组20轮。两组都读取原 stage1 epoch17 的 best.pt；B 不读取 A 的训练结果。原项目源码和权重不覆盖。安装新增 ga_module_probe 包和两份 YAML。

## 设置

- 主干、Neck、检测头参数冻结，所有 BN 统计冻结。
- 训练原对齐、BHLR 和引导融合层。冻结检测头仍保留反向计算图，检测损失可以更新上游模块。
- Mono=KD=0，不执行 PCGrad；保留原检测损失与对齐监督。
- A：新增支持图反馈 eta=0；B：前5轮 eta=0.02/0.04/0.06/0.08/0.10，此后0.10。
- 已对齐 TIR 特征进入 BHLR 的原有梯度路径保持1.0；直接 Add 路径也保留。A并非关闭对齐模块或其全部检测梯度。
- 单卡，默认GPU5；batch4、累积8，有效批量约32。对齐学习率1e-4、BHLR和引导融合层3e-4，余弦降至初始的0.1倍。

## 安装与运行

在YOLO-tcy_v5项目根目录解压压缩包后：

```bash
python GA_ModuleProbe_E17/install.py --repo-root . --device 5 --dry-run
python GA_ModuleProbe_E17/install.py --repo-root . --device 5
python -m ga_module_probe.train --cfg configs/experiments/ga_module_probe/A_support_off.yaml --preflight-only
python -m ga_module_probe.train --cfg configs/experiments/ga_module_probe/B_support_on.yaml --preflight-only
```

检查通过再运行：

```bash
mkdir -p logs
python -u -m ga_module_probe.train --cfg configs/experiments/ga_module_probe/A_support_off.yaml > logs/e17_A_support_off.log 2>&1
python -u -m ga_module_probe.train --cfg configs/experiments/ga_module_probe/B_support_on.yaml > logs/e17_B_support_on.log 2>&1
```

推荐在tmux中顺序执行。只选择空闲GPU，安装时可修改 --device。不要同时在同一张卡上跑两组。

## 检查与结果

预检查在内存中做一次真实优化步骤后恢复模型，不改源权重。B要求检测损失经新增支持路径改变对齐梯度，否则停止；A不要求该增量非零。两组仍要求模块收到检测梯度，且冻结参数/缓冲区在预检查和正式训练中保持不变。

预检查单个真实批次失败时，保留报错检查原因；不要跳过检查，也不能由一个批次直接认定整个模型不可训练。

训练先重新评估完整val基线，预期接近58.65%，以实际统一评估值为准。输出：runs/ga_module_probe/E17_A_support_off 和 E17_B_support_on。

- baseline_val.json：各自训练前基线；两组应一致或在数值误差内。
- preflight.json：真实梯度/冻结检查。
- history.json：逐轮验证指标。
- summary.json：最佳轮次、相对基线增益（百分点）。
- weights/best.pt：包含起点的最佳模型；若没提升，仍为起点。
- weights/best_trained.pt：仅在20轮训练结果中选出的最佳模型，即使低于起点也会保存。
- source_metadata.json：源权重哈希；两组应一致。

中断续训示例：
python -m ga_module_probe.train --cfg configs/experiments/ga_module_probe/A_support_off.yaml --resume runs/ga_module_probe/E17_A_support_off/weights/last.pt

模块阶段权重用本包验证入口：
python -m ga_module_probe.val --weights runs/ga_module_probe/E17_B_support_on/weights/best.pt --split test --device 5 --batch 4

## 验证范围

已完成语法、A/B配置一致性、零反馈配置、缩放梯度测试；使用保存的原版BHLR和引导融合类、合成输入验证前向不变及梯度增量比例。尚未访问服务器实际源码、epoch17权重或UAV-CB数据。真实模型兼容性由服务器上的预检查验证；不保证性能提升。
