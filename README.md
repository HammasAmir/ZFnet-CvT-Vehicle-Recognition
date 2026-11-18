# Cross-Domain Vehicle Classification & Transfer Learning (ZFNet vs. CvT-13)

A PyTorch computer vision benchmark evaluating domain shift and transfer learning between fine-grained vehicle datasets: **US Cars (196 classes)** and **Pakistani Cars (48 classes)**[cite: 1, 2, 3].

## Overview
Vehicle Make and Model Recognition (VMMR) models trained in one geographic region often struggle in another due to differences in vehicle fleets, camera angles, and image resolution[cite: 3, 6]. This project compares a classical CNN (**ZFNet**, 2014) against a hybrid Convolutional Vision Transformer (**CvT-13**, 2021) across direct zero-shot transfer and target-domain adaptation[cite: 1, 2, 3].

### Architectures Compared
| Model | Type | Parameters | Initialization | Optimizer |
| :--- | :--- | :--- | :--- | :--- |
| **ZFNet** (BatchNorm variant) | Pure CNN | ~58.5M – 59.1M[cite: 4] | Trained from scratch[cite: 2, 5] | SGD (Nesterov, LR `0.01`)[cite: 2, 5] |
| **CvT-13** (`microsoft/cvt-13`) | Conv + Transformer | ~19.6M – 19.7M | ImageNet-1k pretrained[cite: 2, 5] | AdamW (differential LRs)[cite: 2, 5] |

---

## Methodologies & Results

### 1. Zero-Shot Cross-Domain Evaluation (`S1_train_US_test_Pak.ipynb`)
Both models are trained exclusively on the 196-class US dataset and evaluated directly on Pakistani images without target-domain training[cite: 1, 6]. Because class labels differ, predicted probabilities are aggregated across the **3 shared makes** (*Suzuki*, *Toyota*, *Honda* — 3,495 test images)[cite: 1, 5, 6].

| Model | Accuracy (%) | Macro-F1 (%) | Macro OvR AUC (%) |
| :--- | :---: | :---: | :---: |
| Majority Baseline (*Always Suzuki*) | 43.30[cite: 6] | — | — |
| **ZFNet** | 50.16[cite: 6] | 46.74[cite: 6] | 65.51[cite: 6] |
| **CvT-13** | **59.69**[cite: 6] | **57.37**[cite: 6] | **73.82**[cite: 6] |

### 2. Target-Domain Adaptation (`zfnet_cvt_cars_Adaptive methodology 1 & 2.ipynb`)
To overcome domain shift, the US-trained backbones are adapted to the 48-class Pakistani dataset (stratified 70/10/20 split) across three regimes[cite: 2, 6]:
- **Baseline:** Trained on Pakistani data without US car pretraining (ZFNet from scratch; CvT-13 from ImageNet weights)[cite: 2, 6].
- **US Linear Probe (`us_linear`):** US-trained backbone frozen; only a new 48-way classification head is trained[cite: 2, 6].
- **US Fine-Tune (`us_finetune`):** End-to-end fine-tuning of the US-trained backbone with a reduced backbone learning rate[cite: 2, 6].

| Model | Adaptation Method | Top-1 Acc (%) | Top-5 Acc (%) | Macro-F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **ZFNet** | Baseline (from scratch)[cite: 2, 6] | 83.12[cite: 6] | 94.55[cite: 6] | 76.86[cite: 6] |
| **ZFNet** | US Linear Probe[cite: 6] | 64.42[cite: 6] | 89.87[cite: 6] | 54.92[cite: 6] |
| **ZFNet** | US Fine-Tune[cite: 6] | **84.42**[cite: 6] | **95.71**[cite: 6] | **77.03**[cite: 6] |
| **CvT-13** | Baseline (ImageNet only)[cite: 2, 6] | **92.60**[cite: 6] | 98.05[cite: 6] | **87.37**[cite: 6] |
| **CvT-13** | US Linear Probe[cite: 6] | 88.05[cite: 6] | 97.40[cite: 6] | 83.85[cite: 6] |
| **CvT-13** | US Fine-Tune[cite: 6] | 91.56[cite: 6] | **98.44**[cite: 6] | 86.61[cite: 6] |

---

## Repository Structure
```text
├── zfnet_cvt_cars_Adaptive methodology 1 & 2.ipynb   # Main pipeline: US training & Pakistani adaptation (baseline, linear probe, fine-tune)
├── S1_train_US_test_Pak.ipynb                        # Zero-shot cross-domain evaluation at shared car-make level
├── requirements.txt                                  # Python dependencies
