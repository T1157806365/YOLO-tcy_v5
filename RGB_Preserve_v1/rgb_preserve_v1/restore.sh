#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$MODEL_DIR/rgb_preserve_v1/manage_install.py" restore "$MODEL_DIR" "$MODEL_DIR"
