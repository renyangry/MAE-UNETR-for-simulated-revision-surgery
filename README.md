# MAE-UNETR for Simulated Revision Surgery

Official implementation accompanying the manuscript:

> **An Automatic Pipeline for Revision Knee Arthroplasty Planning**  
> *Under review — Bone Joint Research*

---

## Overview

This repository provides the source code for an automatic deep learning pipeline designed to support pre-operative planning in revision total knee arthroplasty (rTKA). The pipeline reconstructs full femoral geometry from CT scans in which the distal femur is partially or entirely obscured — a scenario that arises when existing implant hardware degrades image quality or when bone stock must be estimated before hardware removal.

The pipeline is validated under two simulated conditions:
- **Distal femoral resection simulation**: the distal femoral region (up to 10% of total femoral length) is removed from the CT volume prior to inference.
- **Implant presence simulation**: a synthetic implant-shaped mask is applied to the femoral CT volume to replicate signal loss from metallic hardware, after which the pipeline performs full femoral reconstruction.

Full methodological details, experimental results, and clinical context are provided in the manuscript. This repository is intended to allow reproduction of the reported experiments.

---

## Pipeline

```
Input CT scan
      │
      ▼
┌─────────────────────┐
│   3D CT Segmentation │  (nnUNet, two-class: background / femur)
└─────────────────────┘
      │  label = 1 (femur mask)
      ▼
┌─────────────────────┐
│  Femur-region Crop   │  Bounding-box crop guided by femur mask
└─────────────────────┘
      │  cropped femoral CT volume
      ▼
┌──────────────────────────────────────┐
│  Reconstruction Network (MAE-UNETR)  │  Transformer-based encoder–decoder
│  · Encoder: self-supervised ViT      │  trained with masked-volume pretext task
│  · Decoder: UNETR-style segmentation │
└──────────────────────────────────────┘
      │
      ▼
Full femoral 3D segmentation
```

---

## Repository Structure

```
MAE-UNETR-for-simulated-revision-surgery/
├── segmentation/          # nnUNet training and inference scripts
├── preprocessing/         # CT resampling, femur-region cropping
├── mae_unetr/             # MAE-UNETR model definition and training
│   ├── models/
│   ├── train.py
│   └── predict.py
├── simulation/            # Distal-resection and implant-mask simulation scripts
├── evaluation/            # Dice, HD95, and volumetric evaluation scripts
├── configs/               # Training configuration files
├── requirements.txt
└── README.md
```

> **Note:** Pre-trained model weights are available upon reasonable request to the corresponding author, pending completion of the journal review process and any applicable intellectual property agreements.

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8 (GPU training and inference)
- [nnUNet v2](https://github.com/MIC-DKFZ/nnUNet)
- monai >= 1.3

A full list of dependencies is provided in [requirements.txt](requirements.txt).

---

## Installation

```bash
git clone https://github.com/renyangry/MAE-UNETR-for-simulated-revision-surgery.git
cd MAE-UNETR-for-simulated-revision-surgery
pip install -r requirements.txt
```

---

## Usage

### 1. CT Segmentation (nnUNet)

```bash
python segmentation/predict.py \
    --input_dir  /path/to/ct_nifti/ \
    --output_dir /path/to/segmentations/ \
    --model_dir  /path/to/nnunet_weights/
```

### 2. Femur-region Crop

```bash
python preprocessing/crop_femur.py \
    --ct_dir   /path/to/ct_nifti/ \
    --seg_dir  /path/to/segmentations/ \
    --out_dir  /path/to/femur_crops/
```

### 3. Reconstruction (MAE-UNETR)

```bash
python mae_unetr/predict.py \
    --input_dir  /path/to/femur_crops/ \
    --output_dir /path/to/reconstructions/ \
    --weights    /path/to/mae_unetr_weights.pth
```

### 4. Simulation (optional)

```bash
# Distal-resection simulation
python simulation/resect_distal.py \
    --input_dir /path/to/femur_crops/ \
    --out_dir   /path/to/simulated/ \
    --fraction  0.10

# Implant-mask simulation
python simulation/apply_implant_mask.py \
    --input_dir /path/to/femur_crops/ \
    --out_dir   /path/to/simulated_implant/
```

---

## Data Availability

The CT dataset used in this study is not publicly released due to patient confidentiality and institutional ethics requirements. Anonymised data may be made available to qualified researchers upon reasonable written request to the corresponding author, subject to institutional review board approval.

---

## Citation

If you use this code, please cite the associated manuscript once published. The citation will be updated here following acceptance.

```
[Citation pending — manuscript under review at Bone Joint Research]
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
