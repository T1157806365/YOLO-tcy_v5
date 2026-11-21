#!/usr/bin/env python3
"""Exact split-wide evaluation + benchmark of unified 01 or 10."""
import argparse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--weights',default=None)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--device',default='0')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--plots',nargs='?',const='yes',default=None)
    p.add_argument('--preview-num',type=int,default=8)
    p.add_argument('--no-flops',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--save-dir',default=None)
    args=p.parse_args()
    from gmls_roi_hybrid.benchmark import run_benchmark
    run_benchmark(args)

if __name__=='__main__':main()
