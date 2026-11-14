# YOLO-tcy_v5 — Clean GMLS-v2 baseline

Only the best GMLS-v2+DMLS correction+Fine Scale+BHLR-v2(P3)+ADD run from v4 is copied.

## What is preserved?
- Source Python files needed by the chosen model and training/evaluation commands.
- Baseline experiment config reconstructed from the TRAINED checkpoint: `configs/experiments/best_baseline.yaml`.
- Original v4 YAML archived at `docs/source_v4_training.yaml` for provenance (it may differ from trained config).
- RGB/TIR dataset YAML copied into `configs/datasets` (original image data stays external).
- Matching YOLO26m pretrained weights found in v4 (see migration manifest).
- FULL experiment folder at `runs/rlsfa_gtga_bhlr_gmls_v2/yolo26m_UAVCB_GMLSv2_DMLS-Correction_Fine_Scale_BHLRv2-P3_ADD_rgb1920_sem640_tir640_seed0_scratch_bs32_r6_bhlrnew_newbest` (best/last weights, logs, and recorded results).

## Verify without training
```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
python verify_v5.py
bash scripts/validate_best.sh 3
```

## Retrain only when intended
```bash
conda activate tcy
cd /mnt/sda/taochangyong/Projects/Model/YOLO-tcy_v5
# Edit configs/experiments/best_baseline.yaml (train.device especially) before training
bash scripts/train_baseline.sh
```

The saved checkpoint metadata is unmodified. Its embedded dataset paths may still point to v4.
The validation helper overrides RGB/TIR YAML so it uses the v5 copies.
Do not delete v4 until strict load and validation metrics match.

## Provenance
See `docs/migration_manifest.json` for SHA256 and copied file paths.
