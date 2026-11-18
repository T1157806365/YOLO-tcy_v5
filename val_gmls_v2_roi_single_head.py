#!/usr/bin/env python3
"""Evaluate the ONE YOLO26m Detect, with prediction-only GMLS ROI proposals."""
import argparse
from gmls_roi_single_head.runner import run_val

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--weights',default=None)
    p.add_argument('--split',default='test',choices=('val','test','train'))
    p.add_argument('--device',default='5')
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--save-dir',default=None)
    p.add_argument('--plots',default=None,help='Optional directory for ROI/guide mosaics')
    p.add_argument('--smoke',action='store_true')
    run_val(p.parse_args())
