# RGB-Preserve v1 — YOLO-tcy_v5 独立实验版

## 做了什么

- **RGB 主路径初始化**：使用现有 RGB-only YOLO26m 640 `best.pt` 的完整 Backbone、Neck、Detect Head，严格加载所有匹配参数；不再只使用官方 COCO 预训练。
- **TIR Backbone 初始化**：使用现有 TIR-only YOLO26m 640 `best.pt`，单独严格加载 Backbone 权重。
- **P3 的零初始化残差融合**：`RGB_P3 + tanh(alpha) * aligned_TIR_P3`，`alpha=0`；P4/P5 不参与融合。**这是一个受控实验设计，不保证 AP 提升。**
- **对齐**：DMLS 41×41 全局搜索，13×13 局部 softargmax，Fine ±2，Scale 保留。没有 GMLS 多候选/路由器。
- **BHLR-v2**：640 控制实验关闭 BHLR（两条模态输入均 640）；1920 版本保留 BHLR-v2 P3，其内置 `gamma_raw=0` 保证初始 RGB 恒等。
- **两阶段**：Stage1 冻结 RGB Backbone+Neck+Head 参数及对应 BN 统计（Detect Head 保持损失所需的训练输出模式），仅训练 TIR/对齐/P3 residual/BHLR；Stage2 冻结 RGB Backbone，放开 RGB Neck+Head，小学习率更新。DDP 两阶段**重新包装**，避免动态冻结造成跨卡梯度异常。
- **验证脚本**：复用原始 `val_rlsfa_gtga_bhlr_v2.py` 中全部指标代码，使用独立 Checkpoint 加载器。checkpoint 保存 `model_state_dict` 与 `config`。

## 部署

把整个 `RGB_Preserve_v1` 文件夹上传至服务器 `YOLO-tcy_v5` 根目录，执行：

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
bash RGB_Preserve_v1/install.sh "$PWD"
```

安装只增加 `rgb_preserve_v1/` 与 `configs/experiments/rgb_preserve_v1/`，不修改/覆盖现有模型源码和训练器。若目标位置已经存在，会自动备份后替换，生成安装清单以支持恢复。

## 第 0 阶段（必做）：RGB-only 输出一致性

```bash
python rgb_preserve_v1/check_identity.py \
  --cfg configs/experiments/rgb_preserve_v1/uavcb_rgb_preserve_640.yaml \
  --device 3 --batch 1 --max-batches 2
```

要看到 `PASS`，误差应接近零。失败**不要训练**。可把 cfg 换为 `..._1920.yaml` 检查原生高分辨率 BHLR 在零初始化时的恒等性（显存占用更高）。

## 单批训练图检查（建议执行）

```bash
python rgb_preserve_v1/check_train_graph.py \
  --cfg configs/experiments/rgb_preserve_v1/uavcb_rgb_preserve_640.yaml \
  --device 3 --batch 2 --phase stage1
```

出现 `PASS` 再正式训练；若任何一项失败，把 traceback 发回来，不要进入 300+200 Epoch 实验。

## 正式训练

先做 640 验证，建议 1920 版本只有在 640 版 Val 优于 RGB-only Val 后再运行。

```bash
mkdir -p logs pid
nohup python -u rgb_preserve_v1/train.py \
  --cfg configs/experiments/rgb_preserve_v1/uavcb_rgb_preserve_640.yaml \
  > logs/rgb_preserve_v1_640.log 2>&1 &
echo $! > pid/rgb_preserve_v1_640.pid
tail -n 5 -f logs/rgb_preserve_v1_640.log
```

YAML 默认双卡 `device: "4,6"`，全局 `batch: 32`，Stage1 50 Epoch + Stage2 100 Epoch，早停 30；目的是尽快验证是否具备增益。要做与既有训练时长接近的对照，可在 YAML 修改为 300 + 200，其他超参数按需求保持一致。请以 Val 选择参数，不要反复调 Test。

## 验证（训练结束后）

```bash
python rgb_preserve_v1/val.py \
 --weights runs/rgb_preserve_v1/yolo26m_UAVCB_RGB640_TIR640_DMLS_Fine_Scale_P3_Residual_2stage/weights/best.pt \
 --split val --device 3 --batch 4 --workers 4 --conf 0.001 --iou 0.70
```

如果想单卡训练，将 YAML 的 `train.device` 改为 `"3"`。第一次训练前请检查 `rgb_preserve.rgb_weights` 与 `rgb_preserve.tir_weights` 两条路径存在。

## 一键恢复

```bash
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
bash rgb_preserve_v1/restore.sh
```

恢复只处理安装清单中的新增文件，已有同名文件会按备份还原。用户修改过的文件额外留在 `.rgb_preserve_v1_recovery/.../user_modified_at_restore/`。**不会删除 runs、logs、pid、任何训练权重，也不会修改原始 `models/`/`engine/` 源码。**

## 边界 / 注意事项

- 仅在本地完成 Python 语法、纯离线安装恢复流程测试；**尚未在用户的 GPU、Ultralytics 版本和原始权重上完成训练**。
- 严格权重匹配可能因 YOLO26m 单模态与当前代码版本不匹配而报错；应修复版本差异，不可静默 `strict=False`。
- `RGB 640` 实验比较的是同一 RGB 评价坐标的单模态性能，并不是完整 1920 RGB-only 检测器的替代。
- 当前 Stage1/Stage2 不支持中断后续训；默认早停，保留阶段权重用于比较。
- 这是一版可验证假设的隔离实验，不是已取得 AP 提升的模型。
