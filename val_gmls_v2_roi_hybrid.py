#!/usr/bin/env python3
"""Drop-in Hybrid validation entrypoint with Params / FLOPs / FPS / all APs.

Model checkpoints are never modified. All outputs go under
<checkpoint run directory>/evaluation/<split>/, overwritten each run.
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--device', default='3')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plots', nargs='?', const='1', default=None,
                        help='Enable RGB ROI/TIR ROI/guide previews. Original path is ignored: stored inside training run.')
    parser.add_argument('--save-dir', default=None,
                        help='Compatibility argument; reports always saved inside training run.')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Inference warmup repeats on first batch; excluded from FPS and AP.')
    parser.add_argument('--preview-num', type=int, default=8)
    parser.add_argument('--no-flops', action='store_true',
                        help='Skip convolution/linear FLOPs estimation.')
    parser.add_argument('--smoke', action='store_true',
                        help='Only evaluate first batch to confirm model and report code.')
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0 or args.warmup < 0:
        parser.error('batch>=1, workers>=0, warmup>=0 required')
    from gmls_roi_onehead_v2.benchmark import run_benchmark
    run_benchmark(args)


if __name__ == '__main__':
    main()
