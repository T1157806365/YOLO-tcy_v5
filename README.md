# AHR2Net 接入原 YOLO-tcy_v5 训练/验证脚本

该补丁**只修改模型工厂** `models/gtga_bhlr_factory.py`，不覆盖
`train_rlsfa_gtga_bhlr_v2.py`、`engine/trainer.py`、`val_rlsfa_gtga_bhlr_v2.py`。
需要先安装之前的 AHR2Net 完整模型源码（`models/ahr2net_scratch_model.py` 和新模块）。

## 1 安装
```
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
unzip -o AHR2Net_Use_Original_Trainer_Patch.zip
python scripts/integrate_ahr2net_legacy.py
```
程序备份原工厂，重复执行不会重复插入。兼容之前运行过 `scripts/register_ahr2net_v5_factory.py` 的情况。

## 2 YAML 约定
```
model:
  variant: ahr2net_v5_scratch
  pretrained: true
train:
  device: "0,6"
  batch: 32
  batch_mode: global
  # 原始 trainer 不支持 accumulate 参数，勿依赖它
  epochs: 300
  optimizer: SGD
  lr0: 0.01
  stage2:
    enabled: true
    epochs: 200
    lr0: 0.001
```
说明：原始训练器将全局 Batch 32 分配给两个进程，每卡 Batch 16。实际显存是否足够仍需单独验证。

## 3 使用原脚本
```
python train_rlsfa_gtga_bhlr_v2.py --cfg configs/experiments/ahr2net/uavcb_ahr2net_yolo26m_official_ddp2_1920_640.yaml

python val_rlsfa_gtga_bhlr_v2.py --weights runs/ahr2net_v5_scratch/<yaml output.name>/weights/best.pt --split test --device 1 --batch 4 --workers 4 --conf 0.001 --iou 0.70 --plots
```
原始训练器保存 `model_state_dict` + `config` 的完整权重，可以由修订后的同一模型工厂重建 AHR 模型。

## 4 复原
如需退回补丁前工厂：先检查备份文件内容，然后执行
```
cp models/gtga_bhlr_factory.before_ahr2net_legacy_integration.py models/gtga_bhlr_factory.py
```
