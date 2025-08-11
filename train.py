#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrOCR Large fine-tuning for metal-engraved serial numbers
- CSV columns: image_path,label
- Preprocess: CLAHE(4,16) + invert if mean<115 + GRAY->RGB  (train/val/test 동일)
- Optional online augmentation (light, Albumentations 2.0.7-safe)
- Custom Seq2SeqTrainer: uses prefix_allowed_tokens_fn for regex-like constrained decoding
- Metrics: CER + Exact Match(EM)
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A  # 2.0.7

from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

# --------------------------
# Utils
# --------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (ca != cb))
            prev = cur
    return dp[lb]

def character_error_rate(pred: str, ref: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return levenshtein(pred, ref) / max(1, len(ref))

# --------------------------
# Preprocess (train/val/test 동일)
# --------------------------
def preprocess_image_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """
    BGR -> GRAY -> CLAHE(4,16) -> invert if mean<115 -> RGB
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 16))
    gray = clahe.apply(gray)
    if float(np.mean(gray)) < 115.0:
        gray = cv2.bitwise_not(gray)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb

# --------------------------
# Light Online Augmentation (Albumentations 2.0.7)
# --------------------------
def build_online_aug() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=3, border_mode=cv2.BORDER_REFLECT_101, p=0.5),  # ±3°
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(std_range=(5/255.0, 18/255.0), mean_range=(0.0, 0.0), p=0.3),
        # 모션/다운업샘플은 필요 시 추가 가능
    ])

# --------------------------
# Dataset
# --------------------------
class SerialNumberDataset(Dataset):
    def __init__(
        self,
        rows: List[Tuple[str, str]],
        processor: TrOCRProcessor,
        is_train: bool,
        use_online_aug: bool,
    ) -> None:
        self.samples = rows
        self.processor = processor
        self.is_train = is_train
        self.aug = build_online_aug() if (is_train and use_online_aug) else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        rgb = preprocess_image_bgr(img_bgr)
        if self.aug is not None:
            rgb = self.aug(image=rgb)["image"]

        pixel_values = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values[0]
        #with self.processor.as_target_processor():
        #    labels = self.processor(label, return_tensors="pt").input_ids[0]
        labels = self.processor(text=label, return_tensors="pt").input_ids[0]
        return {"pixel_values": pixel_values, "labels": labels}

@dataclass
class DataCollatorForOCR:
    processor: TrOCRProcessor
    decoder_start_token_id: int  # ← 추가

    @staticmethod
    def _shift_right(labels_pad: torch.Tensor, pad_id: int, start_id: int) -> torch.Tensor:
        """
        labels_pad: [B, L] (pad_id로 패딩된 라벨, -100 처리 전)
        return: decoder_input_ids [B, L]
        """
        dec_in = labels_pad.clone()
        dec_in = torch.roll(dec_in, shifts=1, dims=1)
        dec_in[:, 0] = start_id
        # -100 자리는 디코더 입력에 pad_id가 있어야 안전
        dec_in[dec_in == -100] = pad_id
        return dec_in

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])  # [B,3,H,W]
        pad_id = self.processor.tokenizer.pad_token_id

        # 1) pad된 라벨(아직 -100 처리 전)
        labels_list = [b["labels"] for b in batch]
        labels_pad = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=pad_id)

        # 2) loss용 라벨: pad→-100
        labels_for_loss = labels_pad.clone()
        labels_for_loss[labels_for_loss == pad_id] = -100

        # 3) decoder_input_ids: shift-right
        decoder_input_ids = self._shift_right(labels_pad, pad_id=pad_id, start_id=self.decoder_start_token_id)

        return {
            "pixel_values": pixel_values,
            "labels": labels_for_loss,
            "decoder_input_ids": decoder_input_ids,
        }


# --------------------------
# Regex-like constrained decoding
#   Template: N-YYYY-MM-[BT]-ddd
#   Month validity: M1 in {0,1}; if M1==0 => M2 in 1..9, if M1==1 => M2 in 0..2
# --------------------------
class SerialPatternConstraint:
    def __init__(self, tokenizer, eos_id: int, allow_chars: Optional[Dict[int, List[str]]] = None):
        """
        tokenizer: processor.tokenizer
        Pattern indices:
          0:'N', 1:'-', 2:'Y',3:'Y',4:'Y',5:'Y', 6:'-', 7:'M1', 8:'M2', 9:'-', 10:'B|T', 11:'-', 12:'d',13:'d',14:'d'
        """
        self.tok = tokenizer
        self.eos_id = eos_id
        self.length = 15  # fixed template length
        self.char_to_ids = self._build_char_to_ids_map()
        self.allowed = allow_chars  # optional override

    def _tok_ids_for_char(self, ch: str) -> List[int]:
        # map char -> token ids that decode exactly to that char
        ids = self.tok.encode(ch, add_special_tokens=False)
        # Some tokenizers might return multiple ids for a single char (rare). Keep all.
        return ids

    def _build_char_to_ids_map(self) -> Dict[str, List[int]]:
        # 우리가 허용할 문자들
        targets = list("N-BT0123456789")
        m = {ch: [] for ch in targets}

        # 토크나이저 vocab 사이즈 얻기
        vocab_size = getattr(self.tok, "vocab_size", None)
        if vocab_size is None:
            vocab_size = len(self.tok.get_vocab())

        # 단일 토큰을 디코딩했을 때 'N' 혹은 ' N' 처럼
        # 앞에 공백이 붙은 경우까지 모두 허용
        for tid in range(vocab_size):
            s = self.tok.decode([tid], skip_special_tokens=True)
            if not s:
                continue
            s_stripped = s.lstrip()
            if len(s_stripped) == 1 and s_stripped in m:
                m[s_stripped].append(tid)

        # 혹시 못 찾은 문자가 있으면 마지막으로 직접 encode fallback
        for ch in targets:
            if not m[ch]:
                ids = self.tok.encode(ch, add_special_tokens=False)
                if not ids:
                    ids = self.tok.encode(" " + ch, add_special_tokens=False)
                m[ch] = ids

        m["<eos>"] = [self.eos_id]
        return m


    def _allowed_chars_at(self, prefix: str) -> List[str]:
        pos = len(prefix)
        if pos >= self.length:
            return ["<eos>"]

        # helpers
        DIGITS = list("0123456789")
        if pos == 0:   return ["N"]
        if pos == 1:   return ["-"]
        if pos in (2,3,4,5): return DIGITS  # YYYY
        if pos == 6:   return ["-"]
        if pos == 7:   return list("01")    # M1
        if pos == 8:   # M2 depends on M1
            m1 = prefix[7] if len(prefix) > 7 else None
            return list("123456789") if m1 == "0" else list("012")
        if pos == 9:   return ["-"]
        if pos == 10:  return list("BT")
        if pos == 11:  return ["-"]
        if pos in (12,13,14): return DIGITS
        return ["<eos>"]

    def __call__(self, batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        # decode current text (skip specials)
        text = self.tok.decode(input_ids.tolist(), skip_special_tokens=True)
        # strip any leading artifacts
        prefix = text.lstrip()
        allowed_chars = self._allowed_chars_at(prefix)
        # If allowing EOS only (length reached)
        if allowed_chars == ["<eos>"]:
            return self.char_to_ids["<eos>"]

        # map chars -> union of token ids
        allowed_ids: List[int] = []
        for ch in allowed_chars:
            ids = self.char_to_ids.get(ch, [])
            allowed_ids.extend(ids)
        # Always allow EOS if we're at end-1 and model wants to finish early
        allowed_ids.extend(self.char_to_ids["<eos>"])
        # dedup
        return list(dict.fromkeys(allowed_ids))

# --------------------------
# Metrics (uses generated sequences)
# --------------------------
class MetricComputer:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):  # HF compat
            preds = preds[0]
        pad_id = self.processor.tokenizer.pad_token_id
        labels = np.where(labels != -100, labels, pad_id)
        pred_texts = self.processor.batch_decode(preds, skip_special_tokens=True)
        label_texts = self.processor.batch_decode(labels, skip_special_tokens=True)
        ems, cers = [], []
        for p, g in zip(pred_texts, label_texts):
            p, g = p.strip(), g.strip()
            ems.append(1.0 if p == g else 0.0)
            cers.append(character_error_rate(p, g))
        return {"em": float(np.mean(ems)) if ems else 0.0,
                "cer": float(np.mean(cers)) if cers else 0.0}

# --------------------------
# Custom Trainer to inject prefix_allowed_tokens_fn
# --------------------------
class ConstrainedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, prefix_allowed_tokens_fn: Optional[Callable]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix_fn = prefix_allowed_tokens_fn

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # identical to parent but add prefix_allowed_tokens_fn to generate()
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        gen_kwargs = {
            "max_length": self.args.generation_max_length,
            "num_beams": self.args.generation_num_beams,
        }
        if self._prefix_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = self._prefix_fn

        generated_tokens = model.generate(
            inputs["pixel_values"], **gen_kwargs
        )
        # Pad generated tokens to max length
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        loss = None
        if has_labels:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss.detach()
        labels = inputs["labels"] if has_labels else None
        return (loss, generated_tokens, labels)

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/trocr-large-printed")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--use_online_aug", action="store_true")
    parser.add_argument("--use_constraint", action="store_true",
                        help="Enable regex-like constrained decoding (N-YYYY-MM-[BT]-ddd)")
    parser.add_argument("--train_fraction", type=float, default=0.8)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load CSV
    import csv
    rows: List[Tuple[str, str]] = []
    with open(args.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # accept either 'image_path' or 'filename' as image column
        img_key = "image_path" if "image_path" in reader.fieldnames else ("filename" if "filename" in reader.fieldnames else None)
        if img_key is None or "label" not in reader.fieldnames:
            raise ValueError("CSV must contain columns: 'image_path' (or 'filename') and 'label'")
        for r in reader:
            rows.append((r[img_key], r["label"]))

    # ---------------- Grouped Split (누수 차단) ----------------
    import re
    from collections import defaultdict

    def group_key_from_path(p: str) -> str:
        """증강 접미사를 떼고 '원본 basename'을 그룹 키로 사용."""
        name = os.path.splitext(os.path.basename(p))[0]
        # 증강에서 붙이는 접미사들: _base, _aug1, _aug2, ... , _bup, _bdown 등
        # 필요하면 여기에 더 추가 (예: _gammaX, _motion 등)
        pattern = re.compile(r'(?:_base|_aug\d+|_bup|_bdown)$')
        # 여러 번 붙어 있을 수 있으니 반복 제거
        while True:
            new = pattern.sub('', name)
            if new == name:
                break
            name = new
        return name

    # 1) 그룹 만들기
    groups = defaultdict(list)
    for img_path, label in rows:
        g = group_key_from_path(img_path)
        groups[g].append((img_path, label))

    # 2) 그룹 단위 셔플 & 분할
    rng = np.random.default_rng(args.seed)
    group_ids = np.array(list(groups.keys()))
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * args.train_fraction)
    n_val   = int((n_groups - n_train) / 2)
    train_g = group_ids[:n_train]
    val_g   = group_ids[n_train:n_train + n_val]
    test_g  = group_ids[n_train + n_val:]

    # 3) 그룹을 행 리스트로 펼치기
    def flatten(group_keys):
        out = []
        for g in group_keys:
            out.extend(groups[g])
        return out

    train_rows = flatten(train_g)
    val_rows   = flatten(val_g)
    test_rows  = flatten(test_g)

    # 4) 누수 검증(겹치면 assert 터짐)
    assert set(train_g).isdisjoint(val_g)
    assert set(train_g).isdisjoint(test_g)
    assert set(val_g).isdisjoint(test_g)

    print(f"[Split by group] groups: train={len(train_g)}, val={len(val_g)}, test={len(test_g)} "
        f" | samples: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")


    # Processor / Model
    processor = TrOCRProcessor.from_pretrained(args.model_name_or_path)
    
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = VisionEncoderDecoderModel.from_pretrained(args.model_name_or_path)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # ✅ TrOCR-large-printed는 EOS 시작이 기본값 (공식 config와 동일)
    model.config.decoder_start_token_id = tok.eos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id

    metrics  = MetricComputer(processor)

    # Constraint fn (optional)
    prefix_fn = None
    if args.use_constraint:
        eos_id = tok.eos_token_id
        prefix_fn = SerialPatternConstraint(tok, eos_id)

    # Training args
    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=100,
        predict_with_generate=True,
        generation_num_beams=5,
        generation_max_length=32,
        remove_unused_columns=False,
        fp16=(torch.cuda.is_available() and not args.no_fp16),
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        label_smoothing_factor=0.1,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to=["none"],
    )


    # ---------------- Datasets ----------------
    train_ds = SerialNumberDataset(
        train_rows, processor,
        is_train=True, use_online_aug=args.use_online_aug
    )
    val_ds = SerialNumberDataset(
        val_rows, processor,
        is_train=False, use_online_aug=False
    )

    # ---------------- Collator / Metrics ----------------
    # (decoder_input_ids 안전하게 넘기고 싶으면 decoder_start_token_id 추가 버전 사용)
    collator = DataCollatorForOCR(
        processor=processor,
        decoder_start_token_id=tok.eos_token_id,  # 중요!
    )
    metrics = MetricComputer(processor)

    # ---------------- Trainer ----------------
    trainer = ConstrainedSeq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,   # ← 여기 이름과 위에서 정의한 변수명이 일치해야 함
        eval_dataset=val_ds,      # ← 동일
        data_collator=collator,
        tokenizer=processor,
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        prefix_allowed_tokens_fn=prefix_fn,
    )

    # Train
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Evaluate on held-out test (if any)
    if len(test_rows) > 0:
        test_ds = SerialNumberDataset(test_rows, processor, is_train=False, use_online_aug=False)
        test_metrics = trainer.evaluate(test_ds)
        print("[Test] EM={:.4f} CER={:.4f}".format(test_metrics.get("eval_em", -1), test_metrics.get("eval_cer", -1)))

    print("[Best]", trainer.state.best_metric, "saved to", args.output_dir)

# --------------------------
# Production inference helper (regex-constrained)
# --------------------------
def constrained_generate(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    image_bgr: np.ndarray,
    num_beams: int = 5,
) -> str:
    """
    One-off inference with the same preprocess and N-YYYY-MM-[BT]-ddd constraint.
    """
    rgb = preprocess_image_bgr(image_bgr)
    pixel_values = processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values.to(model.device)
    tok = processor.tokenizer
    eos_id = tok.eos_token_id
    prefix_fn = SerialPatternConstraint(tok, eos_id)
    with torch.no_grad():
        out = model.generate(
            pixel_values,
            num_beams=num_beams,
            max_length=32,
            prefix_allowed_tokens_fn=prefix_fn,
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()

if __name__ == "__main__":
    main()
