# nepes-ocr (TrOCR-based Serial Number OCR)

Fine-tuning **microsoft/trocr-large-printed** for metal-engraved serials.

## Data
- CSV: `image_path,label`
- Preprocess: CLAHE(4,16) → invert if mean<115 → GRAY→RGB
- Grouped split to avoid leakage (same base image & its augs go to the same split)

## Train
```bash
export TOKENIZERS_PARALLELISM=false
python train.py \
  --csv_path ./0808aug.csv \
  --output_dir ./0811_trocr_large \
  --model_name_or_path microsoft/trocr-large-printed \
  --num_train_epochs 15 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --accum 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --eval_steps 300 \
  --save_steps 300 \
  --use_online_aug \
  --use_constraint


설명:
- 첫 줄 `\`\`\`` → README 안의 ```bash 코드블록 닫기
- 둘째 줄 `EOF` → heredoc 종료(맨 앞에 공백/탭 있으면 안 됨)

혹시 꼬였으면 `Ctrl+C`로 취소하고 다시 아래처럼 실행해도 됩니다:

```bash
cat > README.md << 'EOF'
# nepes-ocr (TrOCR-based Serial Number OCR)

Fine-tuning **microsoft/trocr-large-printed** for metal-engraved serials.

## Data
- CSV: `image_path,label`
- Preprocess: CLAHE(4,16) → invert if mean<115 → GRAY→RGB
- Grouped split to avoid leakage (same base image & its augs go to the same split)

## Train
```bash
export TOKENIZERS_PARALLELISM=false
python train.py \
  --csv_path ./0808aug.csv \
  --output_dir ./0811_trocr_large \
  --model_name_or_path microsoft/trocr-large-printed \
  --num_train_epochs 15 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --accum 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --eval_steps 300 \
  --save_steps 300 \
  --use_online_aug \
  --use_constraint

\
