# Cross-Domain Vehicle Classification & Transfer Learning (ZFNet vs. CvT-13)

A PyTorch computer vision benchmark evaluating domain shift and transfer learning between fine-grained vehicle datasets: **US Cars (196 classes)** and **Pakistani Cars (48 classes)**.

## Overview
Vehicle Make and Model Recognition (VMMR) models trained in one geographic region often struggle in another due to differences in vehicle fleets, camera angles, and image resolution. This project compares a classical CNN (**ZFNet**, 2014) against a hybrid Convolutional Vision Transformer (**CvT-13**, 2021) across direct zero-shot transfer and target-domain adaptation.

### Architectures Compared
| Model | Type | Parameters | Initialization | Optimizer |
| :--- | :--- | :--- | :--- | :--- |
| **ZFNet** (BatchNorm variant) | Pure CNN | ~58.5M – 59.1M | Trained from scratch | SGD (Nesterov, LR `0.01`) |
| **CvT-13** (`microsoft/cvt-13`) | Conv + Transformer | ~19.6M – 19.7M | ImageNet-1k pretrained | AdamW (differential LRs) |

---

## Methodologies & Results

### 1. Zero-Shot Cross-Domain Evaluation (`S1_train_US_test_Pak.ipynb`)
Both models are trained exclusively on the 196-class US dataset and evaluated directly on Pakistani images without target-domain training. Because class labels differ, predicted probabilities are aggregated across the **3 shared makes** (*Suzuki*, *Toyota*, *Honda* — 3,495 test images).

| Model | Accuracy (%) | Macro-F1 (%) | Macro OvR AUC (%) |
| :--- | :---: | :---: | :---: |
| Majority Baseline (*Always Suzuki*) | 43.30 | — | — |
| **ZFNet** | 50.16 | 46.74 | 65.51 |
| **CvT-13** | **59.69** | **57.37** | **73.82** |

### 2. Target-Domain Adaptation (`zfnet_cvt_cars_Adaptive methodology 1 & 2.ipynb`)
To overcome domain shift, the US-trained backbones are adapted to the 48-class Pakistani dataset (stratified 70/10/20 split) across three regimes:
- **Baseline:** Trained on Pakistani data without US car pretraining (ZFNet from scratch; CvT-13 from ImageNet weights).
- **US Linear Probe (`us_linear`):** US-trained backbone frozen; only a new 48-way classification head is trained.
- **US Fine-Tune (`us_finetune`):** End-to-end fine-tuning of the US-trained backbone with a reduced backbone learning rate.

| Model | Adaptation Method | Top-1 Acc (%) | Top-5 Acc (%) | Macro-F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **ZFNet** | Baseline (from scratch) | 83.12 | 94.55 | 76.86 |
| **ZFNet** | US Linear Probe | 64.42 | 89.87 | 54.92 |
| **ZFNet** | US Fine-Tune | **84.42** | **95.71** | **77.03** |
| **CvT-13** | Baseline (ImageNet only) | **92.60** | 98.05 | **87.37** |
| **CvT-13** | US Linear Probe | 88.05 | 97.40 | 83.85 |
| **CvT-13** | US Fine-Tune | 91.56 | **98.44** | 86.61 |

---

## Repository Structure
```text
├── zfnet_cvt_cars_Adaptive methodology 1 & 2.ipynb   # Main pipeline: US training & Pakistani adaptation (baseline, linear probe, fine-tune)
├── S1_train_US_test_Pak.ipynb                        # Zero-shot cross-domain evaluation at shared car-make level
├── requirements.txt                                  # Python dependencies
