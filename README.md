# nepes-ocr (TrOCR-based Serial Number OCR)

Fine-tuning **microsoft/trocr-large-printed** for **metal-engraved serial number recognition**.  
Developed from **March 2025 ~ Present** to improve OCR accuracy on curved, reflective, and low-contrast serial numbers in semiconductor carriers.

---

## 📅 Development Timeline

### **2025.03 ~ 2025.04 — Baseline & Data Prep**
- Selected **TrOCR** (Transformer-based OCR) for printed text as base model.
- Built initial dataset from carrier images, manually cropped serial regions.
- Applied **CLAHE**, grayscale conversion, inversion for dark-on-bright consistency.
- Expanded tokenizer to include special serial number tokens.
- Evaluated baseline performance: ~85% word accuracy on raw data.

### **2025.05 — Augmentation & LoRA Fine-tuning**
- Added heavy **visual augmentation**:
  - Rotation, ShiftScaleRotate, Brightness/Contrast, Shadow, Noise
- Integrated **LoRA** (PEFT) to fine-tune only decoder projection layers.
- Achieved **99%+ word accuracy** on augmented synthetic + real mix.
- Built CER (Character Error Rate) + Word Accuracy tracking in training loop.

### **2025.06 — Hard Example Mining**
- Collected OCR confusion cases (B↔T, digit swaps).
- Created **hard-example dataset** for targeted fine-tuning.
- Improved robustness against glare and low-contrast edges.

### **2025.07 — Inference Optimization**
- Exported model to **ONNX FP16** with dynamic shapes.
- Optimized inference with **TensorRT** for Jetson Orin AGX deployment.
- Reduced latency from ~1.2s → ~300ms per image.

### **2025.08 — Large-scale Training & Evaluation**
- Consolidated full augmented dataset (real + synthetic).
- Fine-tuned **trocr-large-printed** on 8K+ samples with online augmentation.
- Added **beam search decoding with constraints** (allowed-char filtering).
- Automated evaluation pipeline (`eval_sn.py`) with CSV logging.

---

## 📂 Data Format

- **CSV file**:  
- **Preprocess pipeline**:
1. CLAHE `(clip=4, tile=16)`
2. Invert if mean pixel < 115
3. Convert GRAY → RGB
- **Grouped split**:  
Same base image & its augmentations → same split (to prevent leakage)

---

## 📊 OCR Dataset & Preprocessing

- **Original**: NS2 45 + Lab 16 = 61  
- **Usable**: 52 (NS2 9 removed due to damaged serials)
- **Serials**: Front & back per carrier → 87 total serial number crops.

1. **Data Expansion**
 - "Double-side extraction" to maximize sample count.

2. **1st Augmentation (Mild)**
 - Preserve legibility:
   - Shift, brightness/contrast, blur, sharpening, CLAHE.

3. **2nd Augmentation (Curriculum)**
 - Train with **gradually stronger augmentation** over epochs.

4. **Synthetic Data Trials**
 - **Char-level crops** → Reconstruct serials → 14.17% accuracy.
 - **Synthetic serial generation from crops** → 69.1% accuracy.
 - **OCR-A font synthetic data** → 37.5% accuracy.
 - **Conclusion**: Original-based augmentation performed best.

---

## 🚀 Training 

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

## 📈 **Evaluation**
python eval_sn.py \
  --csv_path ./test.csv \
  --model_path ./0811_trocr_large \
  --beam_width 5 \
  --constraint
