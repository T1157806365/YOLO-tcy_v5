#!/usr/bin/env python3
"""Full val/test global, local and merged detection performance."""
import argparse
from global_local_scratch.runner import run_val

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cfg',required=True)
    p.add_argument('--weights',default=None)
    p.add_argument('--split',default='test',choices=['val','test','train'])
    p.add_argument('--device',default='5')
    p.add_argument('--batch',default=1,type=int)
    p.add_argument('--workers',default=4,type=int)
    p.add_argument('--save-dir',default=None)
    p.add_argument('--smoke',action='store_true')
    run_val(p.parse_args())
