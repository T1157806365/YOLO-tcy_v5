# YOLO-tcy_v5 — BHLR Dual-HR 全部代码及 100 Epoch 消融

**基于已核对的 GitHub 分支**：`T1157806365/YOLO-tcy_v5` / `backup-20261117-no-teacher` (`704dd431...`)

## 1. 修改内容与严格不修改范围

本扩展专门解决第一阶段高分辨率残差构建及高分辨率结构表示：

- Control：现有 `BHLR-v2`，1920 RGB 减独立 LetterBox 640 RGB 双线性上采样；与现有 HVAD/GTGA/TriGate 一致，扩展不注册。
- Clean：由当前 1920 RGB 使用 `area` 下采样生成内部 640 再双线性上采样，构建严格配对的重建残差，沿用原 BHLR `detail_stem`。
- Dual：Clean + 第二路 **原始 1920 RGB 的 PixelUnshuffle→StructureEncoder**，用 `F_HR=F_R+gamma*gate*F_S` 融合。`gamma=0` 初始严格退化为 Clean。
- Guided：Dual + GTGA 当前 `rgb_target_support/aligned_tir_target_support/original_joint_target_support` 三通道指导。门控保留 RGB-only 支持，TIR 不一致不会强制关闭结构分支。

整个方案**不改 YOLO26m-P2、GTGA/GMLS-v2、主 ADD/TriGate、HVAD 的已有结构和损失函数**；只增加模型实例级可选包装、新模块和独立 YAML。原 `models/modules/bhlr_v2.py` 不会被编辑；原 `configs/experiments/gmls_v2/uavcb_gmls_v2_yolo26m_1920_640.yaml` 不会被编辑。

### 训练/推理关系

新模块注册到 `model.bhlr.dual_hr`，其可训练参数纳入 optimizer、DDP 和 checkpoint。

- 在 BHLR 的 `build_lost_detail_bank()` 阶段计算 Clean 残差和原始结构特征；
- 在 `enhance_layer()` 阶段读取现有 GTGA 指导并计算最终融合细节；
- `HVAD` 的 `_detail_bank` 更新为同一最终融合细节，而不是保留旧残差特征；
- 训练监控以 `dualhr_*` 字段进入既有训练 `history_stage1.json`。

当前实现严格要求 **BHLR 仅启用 P3**。如果改为多尺度需先实现共享结构特征跨尺度调用与缓存管理。

## 2. 文件目录

```text
v5_bhlr_dual_hr_patch/
├── install_bhlr_dual_hr.py
├── models/modules/bhlr_dual_hr.py
├── configs/experiments/bhlr_dual_hr/
│   ├── uavcb_hvad_control_officialPT_100e.yaml
│   ├── uavcb_hvad_clean_officialPT_100e.yaml
│   ├── uavcb_hvad_dual_officialPT_100e.yaml
│   └── uavcb_hvad_guided_officialPT_100e.yaml
├── scripts/
│   ├── check_bhlr_dual_hr.py
│   ├── diag_hr_residual.py
│   └── compare_bhlr_dual_hr.py
├── tests/
│   ├── test_dual_hr_mock.py
│   └── test_installer_mock.py
└── README_完整使用说明.md
```

随包附带四份可读 YAML；**安装器会从服务器当前的原始 HVAD YAML 重新生成对应版本**，以保留你本地完整参数/路径及已有自定义配置。新 YAML 放入项目的 `configs/experiments/bhlr_dual_hr/`。

## 3. 安装

```bash
conda activate tcy
cd ~/Projects
unzip -o v5_bhlr_dual_hr_complete.zip
python v5_bhlr_dual_hr_patch/install_bhlr_dual_hr.py \
  --target /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5

cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python scripts/check_bhlr_dual_hr.py
```

安装器先验证实际 v5 文件和工厂锚点；如匹配失败，会报错而不是修改未知结构。原工厂自动备份；重复安装幂等。旧 `.pt` 以及 `runs/` 不会被覆盖。

### 真实模型 Smoke Test

```bash
python scripts/check_bhlr_dual_hr.py --build --mode guided --device 5 --semantic-imgsz 256
```

此步构造模型并在 GPU 上前向（RGB-high=768、RGB-semantic/TIR=256）；它不等于已经验证完整 CUDA DDP 训练，建议先通过再启动 100 Epoch。

## 4. 100 Epoch：直接用 configs/experiments 里的 YAML

均为 **双 GPU 5,6、global batch=24、100 Epoch、SGD lr0=0.01、AMP True、官方 YOLO 预训练**。**不载入历史 HVAD best.pt、不使用 teacher/KD、不执行第二训练阶段**。

在项目根目录：

```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
mkdir -p logs pid

# 首先跑 Guided（4 组应顺序执行，不能并行占用相同 GPU）
nohup python -u train_rlsfa_gtga_bhlr_v2.py \
  --cfg configs/experiments/bhlr_dual_hr/uavcb_hvad_guided_officialPT_100e.yaml \
  > logs/UAVCB_DualHR_Guided_100e.log 2>&1 &
echo $! > pid/UAVCB_DualHR_Guided_100e.pid
tail -f logs/UAVCB_DualHR_Guided_100e.log
```

其他三组分别替换：

```text
configs/experiments/bhlr_dual_hr/uavcb_hvad_control_officialPT_100e.yaml
configs/experiments/bhlr_dual_hr/uavcb_hvad_clean_officialPT_100e.yaml
configs/experiments/bhlr_dual_hr/uavcb_hvad_dual_officialPT_100e.yaml
```

**如显存不足，四组 YAML 的 global batch 一起改为 16 或 8，不能只修改一组。** 注意四组会生成不同的新 `runs/bhlr_dual_hr/` 子目录；避免同一配置重复训练导致覆盖同名目录。

## 5. 验证（独立评估）

```bash
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python val_rlsfa_gtga_bhlr_v2.py \
  --weights runs/bhlr_dual_hr/yolo26m_UAVCB_P2_HVAD_GuidedDualHR_100e_seed0/weights/best.pt \
  --split val --device 5 --batch 4 --workers 4 --conf 0.001 --iou 0.70
```

验证结果优先看 **val AP75、AP50:95**，在确定模型后最后使用 test。历史 55.38% 是 test 分数，不要与新模型 val 直接比较。

## 6. 检查模块是否真正工作

```bash
python scripts/compare_bhlr_dual_hr.py
python scripts/diag_hr_residual.py \
  --cfg configs/experiments/bhlr_dual_hr/uavcb_hvad_guided_officialPT_100e.yaml \
  --split val --limit 100
```

历史文件中的字段：

- `dualhr_gamma`：全局学习到的结构注入强度，初始为 0。
- `dualhr_gate`：结构细节被选择的平均空间门控。
- `dualhr_delta_l1`：真正注入到原 BHLR 细节中的变化幅度。长期近零表示双路径几乎未发挥作用。
- `dualhr_residual_l1`：原残差细节编码的平均绝对激活。
- `dualhr_structure_l1`：原始高分辨率结构编码的平均绝对激活。
- `dualhr_prior`：Guided 的平均可靠性先验。Clean 为 0、Dual 为 1。

残差诊断的 GT vs 背景绝对幅度 **不是检测质量指标**；它只是检查新旧残差构建的数值差异。

## 7. 兼容性与测试边界

- 原 GMLS-v2 YAML 未修改，默认不会启用 `bhlr_dual_hr`。
- 原 BHLR-v2 源码、GTGA 源码、HVAD 模块和原验证入口未修改；原 checkpoint 原样加载由旧配置重建的模型。
- 新实验 checkpoint 的配置中包含 `bhlr_dual_hr`，需保留安装后的工厂与模块才能严格加载。
- 已完成本地 CPU mock：Clean/Dual/Guided 前后向、零增益身份、HVAD 最终细节交接、CPU bf16 autocast、安装器幂等和四组配置生成。
- **没有在你的 GPU 环境执行 YOLO26m-P2 完整 DDP / 100e；精度收益尚未验证。**

如果你已安装过此前 Spatial-HVAD 补丁，代码原则上支持该 HVAD 的 detail bank 接口；但其可能改变原 Control 的行为。为了科学对照，务必在同一个当前 v5 代码版本上完成四组训练，并记录本地 `git diff`。
