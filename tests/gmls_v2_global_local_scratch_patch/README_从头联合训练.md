# GMLS-v2 Global-Local Scratch：从头联合训练完整补丁

## 0. 本版与你最初模型的关系

**以原配置** `configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml` **为基础**，保留 `YOLO26m + GMLS-v2/DMLS + GTGA + BHLR-v2 (P3) + ADD`。新增在线局部原图裁剪、TIR 同步几何采样、第二个 RGB-T 双 Backbone、检测框还原和质量感知融合。**不依赖 HVAD、Tri-Gate，不使用原任何 best.pt，不预先导出 ROI 数据**。

这个补丁**不会修改原配置或核心源码**，而是在相同 `configs/experiments/gmls_v2/` 目录下新增新 YAML。

## 1. 训练逻辑（一次训练，非两阶段）

每次迭代，从 `RSDTPairedDataset` 获取 `RGB_high 1920`、`RGB_semantic 640`、`TIR 640`，然后：

1. Global 网络在 RGB_semantic 与 TIR 上执行 GTGA/GMLS-v2 对齐、BHLR 高分辨率残差 P3 注入、ADD 融合和 YOLO26m 检测，计算原来 GMLS-v2 全局检测损失以及对齐损失。
2. 从本次 global 前向的 `alignment_info.coarse_anchor` 推导预测裁剪中心，同时将 GT 中心的随机偏移裁剪作为**仅用于训练的增强样本**（topk=2、gt_crop_prob=0.7）。这不是 GT 在推理时选框，不是预训练或教师网络。局部裁剪是离散的，不对裁剪中心反向传播；GTGA 仍由原 alignment loss 训练。
3. 从 RGB_high 获取原生 640×640 RGB ROI，并利用相同视野和 GTGA 的采样 `sampling_scale`、`sampling_translation` 对全局 TIR 执行 `grid_sample`，得到局部 640×640 TIR ROI。通过 RGB-high/semantic 的**两路独立 letterbox ratio/pad** 做坐标换算。
4. 局部 RGB-T 双 Backbone（YOLO26m、P3/P4/P5 ADD）执行前向，计算局部检测损失。
5. 总损失 `L = L_global_det + L_alignment + lambda_local * L_local_det`；两个检测网络在**同一个 optimizer** 中从随机初始化训练 100 Epoch，配置 `model.pretrained: false`、`model.global_local.local_pretrained: false`、`train.resume: null`。

测试：Global 分支输出全局检测框，依次为 Top-2 检测候选截取原始 RGB ROI、TIR 同步重采样；当无全局候选时使用 GTGA 粗中心兜底。Local 重新检测，框还原到全局 RGB-semantic 640 坐标后进行保守融合。**测试中不读取 GT 来定位 ROI。**

## 2. 安装

把 `gmls_v2_global_local_scratch_complete.zip` 上传服务器后：

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
unzip -o /path/to/gmls_v2_global_local_scratch_complete.zip -d /tmp
python /tmp/gmls_v2_global_local_scratch_patch/install_gmls_v2_global_local_scratch.py \
  --target /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
```

不会覆盖原 GMLS-v2 YAML；需要重复安装更新此补丁自身文件时可带 `--overwrite`。

## 3. 本地代码检查与服务器 Smoke Test

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python tests/test_gmls_v2_global_local_geometry.py
python train_gmls_v2_global_local_scratch.py \
  --cfg configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640_global_local_scratch.yaml \
  --device 5 --smoke
```

Smoke 只做一次前向/反向/参数更新，不生成权重。只有 Smoke 通过才启动完整训练。

## 4. 从头联合训练 100 Epoch（GPU 5,6）

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
mkdir -p logs pid
CUDA_VISIBLE_DEVICES=5,6 nohup torchrun \
  --standalone --nproc_per_node=2 \
  train_gmls_v2_global_local_scratch.py \
  --cfg configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640_global_local_scratch.yaml \
  > logs/UAVCB_GMLSv2_GlobalLocalScratch_100e.log 2>&1 &
echo $! > pid/UAVCB_GMLSv2_GlobalLocalScratch_100e.pid
tail -f logs/UAVCB_GMLSv2_GlobalLocalScratch_100e.log
```

默认 world=2，global microbatch=2（每 GPU 1），accumulate=16（有效 batch≈32），100 Epoch。联合网络的显存大于原 GMLS-v2，如果单 GPU 1 张仍然 OOM，需要再使用激活检查点或调整模型，而不是盲目增大 batch。

每个训练 epoch 输出 global detection、alignment、local detection、正 ROI 比例。Epoch 1、每 5 Epoch 和最终 Epoch 在 val 集上评估完整三种结果，**best.pt 由 fused val AP50:95 选择**，防止把 Global-only AP 当成完整模型 AP。

> 为了避免此前多卡 rank1 等待验证时发生 600s NCCL 超时，进程组使用 3 小时 timeout；验证由 rank0 在原始（非 DDP）模型上执行，其他 rank 等待。真实服务器上仍需进行双卡 Smoke Test。

## 5. 测试集完整验证

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python val_gmls_v2_global_local_scratch.py \
  --cfg configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640_global_local_scratch.yaml \
  --split test --device 2 --batch 1 --workers 4
```

如要显式指定训练获得的 checkpoint，可以增加 `--weights /full/path/to/best.pt`。输出 `global`、`local`、`fused` 三组 P/R/AP50/AP75/AP50:95、ROI 数量、框替换数、添加数。JSON 保存到 `runs/gmls_v2_global_local_evaluation/test_metrics.json`。

模型输出：

```text
runs/rlsfa_gtga_bhlr_gmls_v2/
  yolo26m_UAVCB_GMLSv2_BHLRv2_GlobalLocalScratch_100e_seed0/
    weights/best.pt
    weights/last.pt
    history.json
```

## 6. 文件清单

- `global_local_scratch/geometry.py`：RGB-high/semantic letterbox 坐标及 TIR geometry grid_sample。
- `global_local_scratch/inference.py`：全局 GT-free TopK/GTGA anchor 兜底的 ROI 生成。
- `global_local_scratch/fusion.py`：局部 box 恢复后的 NMS 与安全替换。
- `global_local_scratch/joint_model.py`：随机初始化全局+局部模型，单次联合 forward/loss/inference。
- `global_local_scratch/runner.py`：训练、保存、完整验证。
- `train_gmls_v2_global_local_scratch.py`：统一从头训练入口。
- `val_gmls_v2_global_local_scratch.py`：测试与消融指标。
- `configs/experiments/gmls_v2/*global_local_scratch.yaml`：从原 GMLS-v2 配置扩展出的 YAML。
- `tests/test_gmls_v2_global_local_geometry.py`：CPU 几何、标签、融合单元测试。

## 7. 需要诚实区分的限制

1. 代码已进行 Python 语法检查、CPU ROI 坐标与标签测试、安装隔离测试；**本环境没有安装用户仓库的 Ultralytics / 无 GPU**，因此没有真实的 1920 GPU forward、YOLO loss、NCCL/DDP 或 100 Epoch 训练结果。必须在用户服务器 Smoke Test 后才可称运行通过。
2. 虽然两套检测器参数从头联合训练，**裁剪位置仍为非可微的离散决策**；不要在论文写“裁剪中心端到端可微”。
3. 训练中一部分 GT 中心裁剪仅用于生成正样本，验证/测试绝不使用 GT 位置；因此验证结果才代表真实粗定位能力。
4. 如果评估指标不升，先查看 global/local/fused AP75、roi_positive 和 `replaced`；不要直接调弱原 global baseline。
5. GTGA 的几何映射来自特征尺度，将其用于像素采样需另外验证局部配准的有效性。
