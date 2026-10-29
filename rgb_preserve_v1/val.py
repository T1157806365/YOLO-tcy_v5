#!/usr/bin/env python3
"""Validate new checkpoint with original evaluator, unmodified."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import val_rlsfa_gtga_bhlr_v2 as original
from rgb_preserve_v1.model import load_rgb_preserve_checkpoint

original.load_model_from_checkpoint = load_rgb_preserve_checkpoint

if __name__ == '__main__':
    original.validate(original.parser().parse_args())
