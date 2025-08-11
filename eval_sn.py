#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, re, argparse
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------- same preprocess as training ----------------
def preprocess_bgr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,16))
    gray = clahe.apply(gray)
    if float(np.mean(gray)) < 115.0:
        gray = cv2.bitwise_not(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# ---------------- grouped split (no leakage) ----------------
def group_key_from_path(p: str) -> str:
    name = os.path.splitext(os.path.basename(p))[0]
    pat = re.compile(r'(?:_base|_aug\d+|_bup|_bdown)$')
    while True:
        new = pat.sub('', name)
        if new == name: break
        name = new
    return name

def make_split(rows: List[Tuple[str,str]], train_fraction: float, seed: int):
    groups = defaultdict(list)
    for p, y in rows:
        groups[group_key_from_path(p)].append((p, y))
    rng = np.random.default_rng(seed)
    gkeys = np.array(list(groups.keys()))
    rng.shuffle(gkeys)
    n_tr = int(len(gkeys) * train_fraction)
    n_rem = len(gkeys) - n_tr
    n_val = max(1, n_rem // 2) if n_rem >= 2 else n_rem
    train_g = gkeys[:n_tr]
    val_g   = gkeys[n_tr:n_tr+n_val]
    test_g  = gkeys[n_tr+n_val:]
    def flatten(keys):
        out = []
        for k in keys: out.extend(groups[k])
        return out
    return flatten(train_g), flatten(val_g), flatten(test_g)

# ---------------- constraint decoding ----------------
class SerialPatternConstraint:
    # N-YYYY-MM-[BT]-ddd  (총 15글자)
    def __init__(self, tok, eos_id):
        self.tok = tok; self.eos_id = eos_id; self.length = 15
        self.map = self._build()

    def _build(self):
        targets = list("N-BT0123456789")
        m = {ch: [] for ch in targets}
        vocab_size = getattr(self.tok, "vocab_size", None) or len(self.tok.get_vocab())
        for tid in range(vocab_size):
            s = self.tok.decode([tid], skip_special_tokens=True)
            if not s: continue
            s2 = s.lstrip()
            if len(s2)==1 and s2 in m: m[s2].append(tid)
        for ch in targets:
            if not m[ch]:
                ids = self.tok.encode(ch, add_special_tokens=False) or self.tok.encode(" "+ch, add_special_tokens=False)
                m[ch] = ids
        m["<eos>"] = [self.eos_id]
        return m

    def _allowed_chars_at(self, prefix: str):
        pos = len(prefix)
        if pos >= self.length: return ["<eos>"]
        DIG = list("0123456789")
        if pos==0: return ["N"]
        if pos==1: return ["-"]
        if pos in (2,3,4,5): return DIG
        if pos==6: return ["-"]
        if pos==7: return list("01")
        if pos==8:
            m1 = prefix[7] if len(prefix)>7 else None
            return list("123456789") if m1=="0" else list("012")
        if pos==9: return ["-"]
        if pos==10: return list("BT")
        if pos==11: return ["-"]
        if pos in (12,13,14): return DIG
        return ["<eos>"]

    def __call__(self, batch_id, input_ids):
        text = self.tok.decode(input_ids.tolist(), skip_special_tokens=True).lstrip()
        allowed = self._allowed_chars_at(text)
        if allowed == ["<eos>"]: return self.map["<eos>"]
        ids = []
        for ch in allowed: ids.extend(self.map.get(ch, []))
        ids.extend(self.map["<eos>"])
        # dedup while preserving order
        seen=set(); out=[]
        for x in ids:
            if x in seen: continue
            seen.add(x); out.append(x)
        return out

# ---------------- metrics ----------------
def cer(a: str, b: str) -> float:
    if len(b)==0: return 0.0 if len(a)==0 else 1.0
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev + (ca!=cb))
            prev = cur
    return dp[lb] / max(1, lb)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--train_fraction", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--out_csv", default="eval_results.csv")
    args = ap.parse_args()

    # load rows
    rows = []
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        img_key = "image_path" if "image_path" in reader.fieldnames else "filename"
        for r in reader:
            rows.append((r[img_key], r["label"]))

    # split
    _, _, test_rows = make_split(rows, args.train_fraction, args.seed)
    print(f"[Eval] test samples = {len(test_rows)}")

    # model / processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.ckpt_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.ckpt_dir).to(device)
    model.eval(); model.config.use_cache = True

    tok = processor.tokenizer
    constraint = SerialPatternConstraint(tok, tok.eos_token_id)

    # eval loop
    rec = []
    with torch.no_grad():
        for pth, gt in test_rows:
            bgr = cv2.imread(pth, cv2.IMREAD_COLOR)
            if bgr is None: continue
            rgb = preprocess_bgr(bgr)
            pv = processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values.to(device)

            # free decoding
            out_free = model.generate(pv, num_beams=args.num_beams, max_length=32)
            pred_free = processor.batch_decode(out_free, skip_special_tokens=True)[0].strip()

            # constrained decoding
            out_con = model.generate(pv, num_beams=args.num_beams, max_length=32,
                                     prefix_allowed_tokens_fn=constraint)
            pred_con = processor.batch_decode(out_con, skip_special_tokens=True)[0].strip()

            gt_s = gt.strip()
            rec.append({
                "image_path": pth,
                "label": gt_s,
                "pred_free": pred_free,
                "em_free": 1 if pred_free==gt_s else 0,
                "cer_free": cer(pred_free, gt_s),
                "pred_constr": pred_con,
                "em_constr": 1 if pred_con==gt_s else 0,
                "cer_constr": cer(pred_con, gt_s),
            })

    # aggregate
    import pandas as pd
    df = pd.DataFrame(rec)
    em_f  = float(df["em_free"].mean()) if len(df)>0 else 0.0
    cer_f = float(df["cer_free"].mean()) if len(df)>0 else 0.0
    em_c  = float(df["em_constr"].mean()) if len(df)>0 else 0.0
    cer_c = float(df["cer_constr"].mean()) if len(df)>0 else 0.0
    print(f"[Free]        EM={em_f:.4f}  CER={cer_f:.4f}")
    print(f"[Constrained] EM={em_c:.4f}  CER={cer_c:.4f}  (beams={args.num_beams})")

    # save CSV
    df.to_csv(args.out_csv, index=False)
    print(f"[Saved] {args.out_csv}")

if __name__ == "__main__":
    main()
