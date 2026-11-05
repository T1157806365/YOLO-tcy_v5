# AG-LHFE v1：对齐引导的局部高分辨率 RGB 特征提取（YOLO-tcy_v5/main）

> 实验代码，**不保证涨点**。当前仅在独立 CPU 特征模块上测试了前向、反向、位置映射和注入关闭；没有在作者 UAV-CB/GPU 环境完成一次 YOLO 端到端运行。请**先 Smoke Test**，再提交训练。

## 1. 核心设计

固定 RGB_high 1920 / RGB_semantic 640 / TIR 640 / YOLO26m、GMLS-v2/DMLS 对齐、RGB-T ADD、P3/P4/P5 Neck，不换框架、不改标注、不改评估器。

- GMLS/DMLS 在 P3 坐标下提供 RGB targetness、**已对齐的 TIR targetness**、joint support；使用已有 GMLS/DMLS quality head 所选择路径的 quality score 作为**可靠性代理**（不是已校准的真实 AlignIoU）。
- **E1 `rgb`**：全部16个候选从 RGB targetness 生成；**E2 `rgb_align`**：8个 RGB 自主候选 + 8个从 RGB/TIR/joint + 估计质量综合的候选，抑制与 RGB 候选重复的位置。总候选数保持16。
- 候选位置 `P3(x,y)` 映射到 1920 RGB 中心，两种真实 RGB 图像块裁剪：48×48 和 96×96 高清像素窗口，均重采样到32×32；**只取局部 patch，不做整幅1920 Backbone**。`grid_sample` 对一个图像的多候选一次采样，避免复制 B×K 张1920输入。
- RGB+Sobel 水平/垂直梯度组成5通道，两尺度浅层 GroupNorm CNN，随后与 P3 RGB / 对齐 TIR 的语义特征融合，预测候选置信度、局部中心偏移和尺寸。
- 对齐后的局部 embedding 使用**可微5×5 Gaussian Splat**映射回 P3。最终 `RGB_P3_new = RGB_P3_old + strength × LocalDetailP3`。`strength = 0.2 × sigmoid(trainable_logit)`，初始化为0.05，`injection_enabled=False` 时准确回到原始 RGB P3。
- `lambda_obj` 和 `lambda_loc` 是训练期候选辅助损失：只用 bbox 监督**已经由图像生成的候选**，GT 绝不直接进入候选提议和推理。候选离 GT 中心不超过2.5个P3像素视作正样本。该设计可能导致训练初期正样本较少，务必监测覆盖率。
- P3 融合后的 Neck/Head、GMLS 几何损失、P4/P5 模态融合完全继承 main。没有另外增加推理检测头。

## 2. 实验对照

四组100轮快速实验（所有实验沿用**main 官方预训练初始化**；不用旧融合 best.pt 初始化；**无 KD / Mono / PCGrad**）：

| YAML | 唯一结构区别 | 目的 |
| --- | --- | --- |
| `E0_OriginalBHLR_100e.yaml` | 原版 GMLS/DMLS + BHLR-v2 | 同训练协议参照 |
| `E1_LocalRGB_100e.yaml` | GMLS/DMLS + RGB候选 + 局部RGB特征 | 验证局部特征形式是否有用 |
| `E2_LocalAligned_100e.yaml` | E1 + 对齐后的TIR候选与质量估计 | 验证对齐是否帮助寻找高分辨率位置 |
| `E3_LocalAligned_NoEdges_100e.yaml` | E2 去掉Sobel分支（用全0代替） | 检查边界/结构特征作用 |

另外提供对应的 `E0/E1/E2_*_300p200.yaml`，与原版300+200训练日程匹配，**只有100e验证有希望后**再执行。旧版历史模型的AP与新实验不能直接做严格的蒸馏因果比较。

训练默认GPU物理编号 `5,6`，`train.batch_mode: global`、`train.batch: 8`，DDP每卡4张，**有效Batch=8**。原始 `engine/trainer.py` 每个batch都会更新，不支持 `accumulate`；不要把这个配置误认为先前蒸馏的有效Batch128。若显存不足，把四份配置里的`train.batch`统一降为4。

## 3. 安装（在你的服务器）

把zip放在项目根目录，再运行：

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5

# 最好先在Git仓库创建独立工作分支；如果这里不是git仓库，可跳过
# git switch -c experiment/ag-lhfe-v1

unzip -o AG_LHFE_v1_Release.zip
python install_ag_lhfe.py --check
python install_ag_lhfe.py
python tests/test_core.py
```

安装仅对 `models/gtga_bhlr_factory.py` **增加一个variant分支**，不会覆盖任何原始网络、训练或评估源码；原 factory 备份为 `models/gtga_bhlr_factory.py.before_ag_lhfe_v1`，可用 `python install_ag_lhfe.py --restore` 恢复。`ag_lhfe/*` 是新文件，旧模型 checkpoint 仍可以原方式严格加载。

**注意**：该补丁按 Github `YOLO-tcy_v5/main` 的工厂代码写成，如果 `--check` 报 marker 不匹配，说明本地factory已有差异，此时**不要强行改**，先核对本地文件。

## 4. 前向+反向 smoke test（先做）

```bash
CFG=configs/experiments/ag_lhfe_v1/E2_LocalAligned_100e.yaml
python train_ag_lhfe.py --cfg "$CFG" --check
python train_ag_lhfe.py --cfg "$CFG" --smoke-test --smoke-device 3
```

期待 `SMOKE PASS`、非零 `ag_lhfe` grad norm，以及 ROI 正例比例 / 几何质量的输出；若显存不足先Smoke CPU不建议使用1920全图，改用其他可用GPU。

## 5. 运行三组独立100轮实验

```bash
mkdir -p logs pid

CFG=configs/experiments/ag_lhfe_v1/E2_LocalAligned_100e.yaml
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  python -u train_ag_lhfe.py --cfg "$CFG" \
  > logs/ag_lhfe_E2_100e.log 2>&1 &
echo $! > pid/ag_lhfe_E2_100e.pid
tail -n 30 -f logs/ag_lhfe_E2_100e.log
```

`train_ag_lhfe.py` 复用项目原本的 DDP auto-launch 和 `engine.trainer.train_experiment`，不会改 checkpoint schema。输出：

`runs/ag_lhfe_v1/yolo26m_UAVCB_E2_LocalAligned_100e_seed0/weights/best.pt`

对照组把 CFG 替换成 E0、E1、E3 对应 YAML，**必须分别独立训练**。先至少完成 E0、E1、E2，再决定是否跑E3。

## 6. 监控两个模块的作用

原训练器 `history_stage1.json` 中新增的字段：

- `lhfe_proposal_gt_recall`：当前训练batch GT中心被topK候选半径2.5覆盖的比例；目标应提高，但不预设必须达到的数值。
- `lhfe_positive_ratio`：正样本候选占比；接近0时表示局部ROI头可能学不到定位。
- `lhfe_injection_rms`：实际P3注入相对于RGB P3的RMS强度，不是AP百分比。
- `lhfe_strength`：可学习的注入幅度（范围0~0.2）。
- `lhfe_geometry_quality`：当前GMLS/DMLS预测质量代理的均值，**不是**真实AlignIoU。
- `lhfe_obj`、`lhfe_loc`、`lhfe_obj_gate_mean`：辅助置信度/偏移/尺寸损失及门控均值。

```bash
python -m ag_lhfe.monitor \
  --run runs/ag_lhfe_v1/yolo26m_UAVCB_E2_LocalAligned_100e_seed0 \
  --last 5
```

不改变任何权重的真实数据候选审计（前96张Val）；要全679张改`--images 0`：

```bash
W=runs/ag_lhfe_v1/yolo26m_UAVCB_E2_LocalAligned_100e_seed0/weights/best.pt
python -m ag_lhfe.audit --weights "$W" --split val --device 3 --batch 2 --images 96
```

监控 `proposal_recall`、`mean_injection_rms_ratio`、`mean_roi_objectness`、`mean_absolute_offset_p3`、`fraction_crossmodal_proposals`。

## 7. 原版验证与同权重注入关闭实验

```bash
W=runs/ag_lhfe_v1/yolo26m_UAVCB_E2_LocalAligned_100e_seed0/weights/best.pt

python val_rlsfa_gtga_bhlr_v2.py \
  --weights "$W" --split val --device 3 --batch 4 --workers 4 --conf 0.001 --iou 0.70

python -m ag_lhfe.eval_probe \
  --weights "$W" --split val --device 3 --batch 4 --workers 4
```

`eval_probe`加载**同一份权重**，分别启用和关闭 `ag_lhfe.injection_enabled`，输出 AP50/AP75/AP50:95 的`on_minus_off`。这是机制审计，并不等价于从头训练“模块关闭”消融。

如果E2比E1更好，但E2 `ON-OFF≈0` 或候选覆盖率很低，则尚不足以证明对齐引导的高分辨率局部细节发挥了独立作用。

## 8. 解释结果与可能的失败模式

1. **E1不比E0好**：先检查局部 patch尺度、proposal GT覆盖率和正例比例；不要先增大strength。
2. **E1提高但E2没有提高**：对齐TIR候选 / GMLS质量选择尚未提供增益，检查RGB和跨模态候选覆盖率、质量预测准确性。
3. **E2提高、ON-OFF也明显为正**：局部高分辨率特征对检测有可测贡献，下一步再做微小目标分组与多种子重复。
4. **OOM/非有限loss**：降低所有对照的全局batch，先关AMP做一次Smoke，检查原对齐损失。不要默默丢弃NaN梯度。
5. 真实推理的patch中心完全由RGB/TIR预测，没有读取GT。所谓质量“reliability”是GMLS质量头估计值，不能当作几何真值。

## 9. 兼容性与限制

- 只支持HBB及项目现有UAV-CB标签格式。
- 只修改P3，P4/P5沿用原双流融合；没有额外全分辨率主干。
- 额外辅助定位头用于训练，但不是推理检测输出。
- 当前实现的TopK候选索引**不可微**（与标准TopK候选检索一致）；局部采样/编码、中心偏移、特征散射与检测loss反向是可微的。GMLS本身通过原有alignment辅助损失学习。
- `get_model_complexity`等基于MAC hooks的统计可能无法精确计入动态grid_sample和scatter操作，务必结合真实GPU FPS/延迟。
- `train_ag_lhfe.py` 复用原生训练器，**没有**KD、Mono、PCGrad、梯度累积、自动断点续训（原本`train.resume`未实现）。这些功能需单独适配，不能通过在YAML里添加字段假装支持。
