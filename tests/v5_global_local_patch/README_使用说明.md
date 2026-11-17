# YOLO-tcy_v5 · RGB-T 全局定位 + 原生 640×640 局部高分辨率精检

适用仓库：`T1157806365/YOLO-tcy_v5`，基于 `backup-20261117-no-teacher`（commit `704dd431`）的真实接口适配。

**目的：** 冻结当前 HVAD-GeoHR v1（test AP50:95≈0.55377）作为第一阶段定位器。根据预测框中心，在 1920 RGB letterbox 输入中裁剪原生 640 RGB ROI；用同一空间 ROI 经 GTGA 几何映射直接从 TIR 640 全图采样出 640 TIR ROI；用新的、单独训练的 RGB-T ADD YOLO26m-P2 对 ROI 检测；还原坐标并保守融合结果。

与之前 LARA 网络不同：**不替换 v5 全局检测网络、不改 GTGA/BHLR/HVAD 的源文件、不加载 GT 到预测中心、不重新对原 YOLO26 End-to-End 输出做全局 NMS。**

## 包含文件

```
global_local/geometry.py             # LetterBox 坐标、TIR GTGA 仿射网格、RGB/TIR 原生 640 裁剪和框还原
global_local/inference.py            # v5 全局 forward 与候选生成
global_local/roi_dataset.py          # 导出的 ROI train/val 数据加载器，复用 v5 loss 格式
global_local/local_model.py          # 使用原 RGBTDetectionModel 的局部 ADD / P2 检测器
global_local/fusion.py               # 保留全局置信度的安全框替换
export_global_local_rois.py          # 从冻结全局模型预测中心生成 ROI 训练集
train_global_local_roi.py           # 新的 ROI 训练脚本，支持单卡和 torchrun DDP
val_global_local.py                 # 完整全局-局部在线推理/三种模式 AP/FPS/覆盖率
preview_global_local_rois.py        # RGB/TIR 裁剪可视化检查
install_global_local.py             # 只复制新增文件，不修改旧代码
configs/experiments/global_local/   # 3套 YAML
```

## 0. 安装

```bash
conda activate tcy
cd ~/Projects
unzip -o v5_global_local_complete.zip
python v5_global_local_patch/install_global_local.py \
  --target /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5

cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python tests/test_geometry_fusion.py  # 仅 CPU 几何与标签测试（若测试文件是从 zip 安装）
```

> 仓库没有 `.git` 也没关系；安装器不会覆盖原文件。若已有同名新增文件，自动备份旧版本。

## 1. 导出 train/val 真实候选 ROI（必须先做）

```bash
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python export_global_local_rois.py \
  --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml \
  --splits train val \
  --device 5
```

输出为：

```
data_generated/global_local_rois_top2/
  train.jsonl, val.jsonl
  images/train/RGB/, images/train/TIR/
  images/val/RGB/, images/val/TIR/
  labels/train/, labels/val/
  train_summary.json, val_summary.json
```

**重要：** train/val 裁剪中心只使用全局模型预测；如果没有检测框，则使用 GTGA 粗锚点作为无标签兜底。不使用 GT 中心。GT 只用于训练 ROI 标签与覆盖率统计。train 额外添加一次非 GT 的随机中心扰动；val 完全不扰动。原数据集不修改。两种 RGB 尺度分别 letterbox，用 metadata 反解坐标，不能简单乘 3。

在生成小规模样例时：

```bash
python export_global_local_rois.py --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml --splits train val --device 5 --max-images 8
```

**注意：** 小规模样例会覆盖现有的 train.jsonl / val.jsonl！真实训练之前必须重新运行不带 `--max-images` 的完整导出。

检查空间配准：

```bash
python preview_global_local_rois.py \
  --roi-root data_generated/global_local_rois_top2 --split val --num 24
```

查看两路图像中无人机是否对应。如果局部 TIR 经 GTGA 对应失败，应该先检查 q = scale*p+translation 和 ROI 的 LetterBox 映射，暂勿开始100e训练。

## 2. 独立训练局部 RGB-T ADD/P2 网络

仍使用原来 `configs/experiments/` 下的 YAML，但训练入口是新脚本（因为数据已经变成 640 ROI，不是原来的双分辨率整图输入）。

先 Smoke Test（正式导出数据以后）：

```bash
python train_global_local_roi.py \
  --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml --smoke
```

单卡正式训练：

```bash
python -u train_global_local_roi.py \
  --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml
```

双卡（物理 GPU 5,6）：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=5,6 nohup torchrun --standalone --nproc_per_node=2 \
  train_global_local_roi.py \
  --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml \
  > logs/UAVCB_GlobalLocal_LocalRGBT_P2_100e.log 2>&1 &
tail -f logs/UAVCB_GlobalLocal_LocalRGBT_P2_100e.log
```

这里是官方 `yolo26m.pt` 初始化两个局部 backbone，局部无 BHLR/GTGA（TIR 已同步几何采样）。训练不修改原来的全局 HVAD `best.pt`。局部最佳权重：

`runs/global_local_roi/uavcb_local_rgbt_add_top2_100e/weights/best.pt`

## 3. 完整全局→局部在线验证

```bash
python val_global_local.py \
  --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml \
  --split val --device 5
```

输出同一批图片下三组：
- `global_only`：原 HVAD 不变（应与原验证 AP 一致，在同一 val、同一预处理/置信度下）
- `local_only`：仅原生 ROI 模型检测，框已恢复到 RGB-semantic 640 坐标
- `safe_replace`：满足 IoU、尺寸、中心距离和置信度门槛时仅替换全局预测框坐标，**不改变全局框得分**；全局完全没有框才允许大于 0.9 的局部框救援

还记录 ROI GT 完整覆盖率、TIR 采样有效区域占比、平均最优 IoU、替换次数与推理 FPS。

对比 Top-1：

```bash
python val_global_local.py \
  --cfg configs/experiments/global_local/uavcb_global_local_top1.yaml \
  --split val --device 5
```

更严格的局部框替换：

```bash
python val_global_local.py \
  --cfg configs/experiments/global_local/uavcb_global_local_strict.yaml \
  --split val --device 5
```

仅在 val 上确定方案以后再跑 test：

```bash
python val_global_local.py \
  --cfg configs/experiments/global_local/uavcb_local_rgbt_add_100e.yaml \
  --split test --device 5
```

## 数据与准确性的边界

1. **GTGA 是特征级估计。** 将其全局仿射映射应用于 TIR 像素采样是本方案的研究假设，不保证像素级配准；实际需要检查 ROI 可视化和 TIR 采样有效率。
2. **不能简单说目标被“凭空放大”。** 1920 原图已有像素没有被缩小，原生 640 裁剪比整图 640 语义输入保留约 3 倍线性像素密度；如果原始采集图分辨率本来就低，收益会受限。
3. 原 v5 真实多分辨率数据以各自 LetterBox 分支生成，所以将变换链严格按 `rgb_ratio_pad` 和 `rgb_semantic_ratio_pad` 还原。
4. 局部模型新训练，**不是复用 HVAD 原有检测头**。全局模型的优异性能由原权重保护，局部性能需实验验证，不能预言 AP 一定提高。
5. 先在 val 调参和选模型、test 仅做最后评估。UAV-CB 数据此前发现跨分割时序序列近邻，建议论文时额外报告时序去重结果。
6. ZIP 中 CPU 几何、标签与融合单测已过；**真实 CUDA/Ultralytics 环境、双卡 DDP、100e 收敛情况尚未在你的机器运行验证**。先使用 `--max-images 8` 导出与 `--smoke` 测试，再完整导出训练。
7. 当前两个检测阶段使用独立参数，第一阶段冻结，不声称端到端可微；待验证实际收益后再考虑端到端候选中心优化。
