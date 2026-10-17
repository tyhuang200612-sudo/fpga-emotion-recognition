"""Create private, human-review-only neutral/sad hard-sample shortlist.
NEVER use TEST, relabel automatically, or confuse misclassification with bad labels.
"""
from __future__ import annotations
import argparse,csv,json,math,random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from tools.quantize_c_int8 import LABELS, load_checkpoint, rows, subset


@torch.inference_mode()
def analyze(model,root,records,split):
    flagged=[]
    for start in range(0,len(records),40):
        samples=records[start:start+40]
        pictures=[]
        for row in samples:
            with Image.open(root/row["path"]) as im:
                pictures.append(np.array(im.convert("L").resize((48,48),Image.Resampling.BILINEAR),copy=True))
        x=torch.from_numpy(np.stack(pictures)).unsqueeze(1).float()/255
        probs=torch.softmax(model(x),1).numpy()
        for row,p in zip(samples,probs):
            label=int(row["label_id"]);chosen=int(np.argmax(p))
            gap=abs(float(p[0]-p[2]))
            confuse=label in (0,2) and chosen in (0,2) and label!=chosen
            flags=[]
            if confuse:flags.append("neutral_sad_confusion")
            if label in (0,2) and gap<0.12:flags.append("neutral_sad_low_margin")
            if p[label]<0.25:flags.append("low_assigned_label_probability")
            if not flags:continue
            flagged.append(dict(split=split,path=row["path"],sha256=row["sha256"],
                label=LABELS[label],predicted=LABELS[chosen],
                label_probability=round(float(p[label]),6),
                neutral_probability=round(float(p[0]),6),
                happy_probability=round(float(p[1]),6),
                sad_probability=round(float(p[2]),6),
                score=round(1-float(p[label])+(0.4 if confuse else 0)+(0.15 if gap<0.12 else 0),5),
                flags=",".join(flags)))
    return flagged


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--manifests",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--train-count",type=int,default=2500)
    a=p.parse_args()
    model=load_checkpoint(a.checkpoint)
    model.eval();torch.set_num_threads(min(4,torch.get_num_threads()))
    train=subset(rows(a.manifests/"train.csv"),a.train_count,7431)
    validation=rows(a.manifests/"val.csv")
    all_results=analyze(model,a.dataset_root,train,"train")+analyze(model,a.dataset_root,validation,"val")
    all_results.sort(key=lambda x:-x["score"])
    a.out.mkdir(parents=True,exist_ok=True)
    fields=["split","path","sha256","label","predicted","label_probability","neutral_probability",
            "happy_probability","sad_probability","score","flags"]
    with (a.out/"human_review_candidates.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(all_results)
    counts=dict(Counter(f for row in all_results for f in row["flags"].split(",")))
    stats=dict(train_sample=len(train),val_count=len(validation),
        candidates=len(all_results),flags=counts,
        auto_relabels=0,auto_deletions=0,test_used=False,
        note="Model disagreement is not proof of annotation error. Review labels manually before changing anything.")
    (a.out/"summary.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n")
    # Sheet is private GitHub artifact; never publish face images in README or logs.
    top=all_results[:72]; side=124
    img=Image.new("RGB",(9*side,math.ceil(len(top)/9)*side),"white")
    draw=ImageDraw.Draw(img)
    for i,item in enumerate(top):
        with Image.open(a.dataset_root/item["path"]) as pic:
            tile=pic.convert("RGB").resize((96,96),Image.Resampling.NEAREST)
        px=(i%9)*side;py=(i//9)*side
        img.paste(tile,(px+12,py))
        draw.text((px+3,py+99),f'{item["label"][:1]}>{item["predicted"][:1]} {item["label_probability"]:.2f}',fill="#202b23")
        draw.text((px+3,py+111),f'{item["split"]} #{item["sha256"][:5]}',fill="#666666")
    img.save(a.out/"PRIVATE_manual_review_sheet.jpg",quality=85)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
