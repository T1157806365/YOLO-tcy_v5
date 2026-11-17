#!/usr/bin/env python3
"""Visual QC of exported RGB and geometrically sampled TIR crops.

Draws local RGB GT boxes only as a diagnostic overlay. Neither ROI generation
nor the local model receives GT positions at inference.
"""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--roi-root',required=True)
    p.add_argument('--split',choices=('train','val','test'),default='val')
    p.add_argument('--num',type=int,default=20)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    root=Path(args.roi_root);file=root/f'{args.split}.jsonl'
    with file.open(encoding='utf-8') as f: rows=[json.loads(x) for x in f if x.strip()]
    rng=random.Random(0)
    rows=rng.sample(rows,min(len(rows),args.num))
    out=Path(args.output or root/f'preview_{args.split}')
    out.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        rgb=cv2.imread(str(root/row['rgb_path']),cv2.IMREAD_COLOR)
        tir=cv2.imread(str(root/row['tir_path']),cv2.IMREAD_GRAYSCALE)
        if rgb is None or tir is None:raise FileNotFoundError(row['id'])
        h,w=rgb.shape[:2]
        for box in row['boxes_xywh']:
            cx,cy,bw,bh=box
            x1=int((cx-bw/2)*w);x2=int((cx+bw/2)*w)
            y1=int((cy-bh/2)*h);y2=int((cy+bh/2)*h)
            cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,0,255),2)
        thermal=cv2.applyColorMap(tir,cv2.COLORMAP_INFERNO)
        label=f"rank={row['candidate_rank']} det_conf={row['candidate_confidence']:.3f} tir_valid={row['tir_valid_fraction']:.3f} boxes={len(row['boxes_xywh'])}"
        panel=cv2.hconcat((rgb,thermal))
        cv2.putText(panel,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
        cv2.imwrite(str(out/f'{i:03d}_{row["id"]}.jpg'),panel)
    print(f'Previewed {len(rows)} paired crops in {out}')
if __name__=='__main__':main()
