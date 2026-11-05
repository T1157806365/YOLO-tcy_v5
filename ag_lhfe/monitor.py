"""Read the latest available epoch metrics from the clean trainer's history JSON."""
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True,help='Path to runs/ag_lhfe_v1/<experiment>')
    p.add_argument('--last',type=int,default=5)
    args=p.parse_args()
    folder=Path(args.run)
    for history_path in (folder/'history_stage1.json',folder/'history_stage2.json'):
        if not history_path.exists():continue
        hist=json.loads(history_path.read_text(encoding='utf-8'))
        if isinstance(hist,dict):hist=hist.get('history',[])
        print('\n',history_path,'rows',len(hist))
        headers=['epoch','map5095','map75','map50','lhfe_proposal_gt_recall',
                 'lhfe_injection_rms','lhfe_strength','lhfe_positive_ratio',
                 'lhfe_obj','lhfe_loc','lhfe_geometry_quality','train_loss']
        print(' '.join(f'{k:>15s}' for k in headers))
        for row in hist[-args.last:]:
            print(' '.join(f'{row[k]:15.5f}' if k in row and isinstance(row[k],(int,float))
                           else ' '*15 for k in headers))
    print('Note: candidate coverage is a training-batch statistic; use audit.py for held-out coverage.')
if __name__=='__main__':main()
