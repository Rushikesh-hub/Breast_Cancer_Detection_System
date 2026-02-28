<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=AI-Enhanced%20Breast%20Cancer%20Diagnosis&fontSize=36&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Mammography%20%E2%9C%A6%20FNAC%20%E2%9C%A6%20Gail%20Risk%20Model&descAlignY=60&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Open%20Source-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

<br/>

> **🏥 Built with RIMS & Bosch Global Software Technologies**
> *An AI-integrated diagnostic system designed to reduce breast cancer mortality through early, accurate, and accessible detection.*

<br/>

---

</div>

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [💡 Motivation & Problem Statement](#-motivation--problem-statement)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [🧠 Modules](#-modules)
  - [Module 1: Gail Risk Screening](#module-1-gail-risk-screening)
  - [Module 2: Mammogram BI-RADS Rating](#module-2-mammogram-bi-rads-rating)
  - [Module 3: FNAC Malignancy Classification](#module-3-fnac-malignancy-classification)
- [📊 Results](#-results)
- [🛠️ Tech Stack](#%EF%B8%8F-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [📁 Dataset Information](#-dataset-information)
- [🔬 Explainability (Grad-CAM)](#-explainability-grad-cam)
- [🔒 Ethics & Privacy](#-ethics--privacy)
- [🔭 Future Scope](#-future-scope)
- [👥 Team & Acknowledgments](#-team--acknowledgments)
- [📚 References](#-references)

---

## 🎯 Overview

Breast cancer is the **second leading cause of cancer-related deaths** among women globally. With approximately **232,000 new diagnoses** annually (US, 2015), early detection is paramount — regular screening can reduce mortality risk by up to **60% within 10 years**.

This project delivers an **end-to-end AI diagnostic pipeline** that:

| Component | Description |
|-----------|-------------|
| 🔬 **FNAC Analysis** | Automated nuclear feature extraction from cytology slides using a human-in-the-loop approach |
| 🩻 **Mammogram BI-RADS Rating** | Transfer learning (VGG-16) for automated classification across all 6 BI-RADS categories |
| 📈 **Risk Screening** | Real-time breast cancer risk scoring via the Gail model |
| 👁️ **Explainability** | Grad-CAM visualizations and GLM transparency for clinical trust |

---

## 💡 Motivation & Problem Statement

Traditional diagnostic workflows face three core challenges:

```
⏱️  TIME-CONSUMING     →  Scheduling delays, multi-step radiologist reviews
🏥  RESOURCE-INTENSIVE  →  Requires specialized equipment and trained professionals
⚠️  HUMAN ERROR        →  Inter-observer variability, especially in dense breast tissue
```

**Dense breast tissue** is particularly problematic — it appears similar in density to tumours on mammograms, leading to false negatives. FNAC, while minimally invasive and cost-effective, can also yield inconclusive results due to low cellularity or complex lesion features.

**Our solution**: Combine AI-driven automation with a *human-in-the-loop* design to assist — not replace — radiologists and pathologists, reducing delays while improving accuracy.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│    Patient Details  │  Mammogram Images  │  FNAC Slides     │
└──────────┬──────────┴────────┬───────────┴────────┬─────────┘
           │                  │                     │
           ▼                  ▼                     ▼
┌──────────────────┐  ┌───────────────┐   ┌──────────────────┐
│  PREPROCESSING   │  │ MAMMOGRAM     │   │  FNAC ANALYSIS   │
│  & SCREENING     │  │ ANALYSIS      │   │  MODULE          │
│                  │  │               │   │                  │
│  • Gail Model    │  │  • VGG-16     │   │  • Cell Boundary │
│  • Risk Score    │  │    Transfer   │   │    Detection     │
│  • Demographic   │  │    Learning   │   │  • Active Contour│
│    Factors       │  │  • End-to-End │   │    Refinement    │
│                  │  │    CNN        │   │  • Feature       │
│                  │  │  • SMOTE      │   │    Extraction    │
└──────────────────┘  └───────┬───────┘   └────────┬─────────┘
                              │                    │
                              ▼                    ▼
                     ┌─────────────────┐  ┌────────────────┐
                     │  BI-RADS Rating │  │ Ensemble Model │
                     │  (0–6 Scale)    │  │ SVM + RF +     │
                     │                 │  │ XGBoost        │
                     └────────┬────────┘  └───────┬────────┘
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                              ┌──────────────────┐
                              │   RESULTS PAGE   │
                              │  • BI-RADS Score │
                              │  • Malignancy %  │
                              │  • Grad-CAM Maps │
                              │  • Risk Report   │
                              └──────────────────┘
```

The system was developed in collaboration with **5 senior doctors** from RIMS and Clarity Advanced Imaging Center, using a **weekly agile-based workflow**.

---

## 🧠 Modules

### Module 1: Gail Risk Screening

The **Gail Model** provides real-time personalized risk assessment using:

- 👤 Age and demographic data
- 👨‍👩‍👧 Family history of breast cancer
- 🔬 Age at menarche and first childbirth
- 📋 Number of prior biopsies

> **Why Gail?** Among COIMBRA, BCSC, and Gail models, the Gail model was selected for its simplicity, reliance on widely available clinical data, and suitability for resource-limited Indian healthcare settings — including rural areas where mammograms and genetic tests are scarce.

---

### Module 2: Mammogram BI-RADS Rating

**Input:** Four-view mammograms (R-CC, L-CC, R-MLO, L-MLO)

**Pipeline:**
1. **End-to-End CNN** — Pre-screens all 4 views simultaneously, trained on 1M+ images, identifies BI-RADS 0 and 1 (non-suspicious)
2. **VGG-16 Transfer Learning** — Fine-tuned on INBreast dataset, assigns specific BI-RADS ratings (4 and 5) for suspicious findings
3. **SMOTE** — Addresses class imbalance for rare malignancy cases
4. **Data Augmentation** — Noise injection and scaling on frozen VGG-16 features to improve generalization

**BI-RADS Scale:**

| Score | Category | Clinical Action |
|-------|----------|-----------------|
| 0 | Incomplete | Additional imaging required |
| 1 | Negative | Routine screening |
| 2 | Benign | Routine screening |
| 3 | Probably Benign | 6-month follow-up |
| 4A/4B/4C | Suspicious | Tissue biopsy |
| 5 | Highly Suspicious | Biopsy strongly recommended |
| 6 | Known Malignancy | Surgical planning |

---

### Module 3: FNAC Malignancy Classification

A fully interactive, **human-in-the-loop** cytology analysis pipeline:

```
Image Load → Pathologist ROI Selection → Cell Boundary Drawing
     → Active Contour Refinement → Feature Extraction
          → Ensemble Classifier → Malignancy Prediction → Export
```

**Extracted Nuclear Features:**

| Feature | Metrics Computed |
|---------|-----------------|
| Area | Mean, SE, Worst |
| Perimeter | Mean, SE, Worst |
| Radius | Mean, SE, Worst |
| Smoothness | Mean, Worst |

Features are **adjusted for microscope magnification** and fed into an ensemble of:
- 🌲 Random Forest
- 📈 Gradient Boosting
- 🔷 Support Vector Machine (SVM)

**Custom RIMS-FNAC Dataset:** 462 patient records collected from RIMS, India — validated against the Wisconsin Breast Cancer Dataset (WBCD) using Kolmogorov-Smirnov tests.

---

## 📊 Results

### FNAC Classification Performance

| Dataset | Accuracy | ROC-AUC | Notes |
|---------|----------|---------|-------|
| WBCD Test Data | **98%** | **1.00** | Near-perfect classification |
| RIMS-FNAC Dataset | **67%** | **0.77** | Cross-dataset generalization |
| FNAC (Top 6 Features) | — | **0.896** | Best FNAC AUC with feature selection |

> The FNAC-based module achieves **98.83% accuracy** (p = 0.049, 95% CI) when evaluated in-distribution, significantly reducing reliance on invasive biopsy.

### Mammogram Suspicious/Non-Suspicious Classification

| Confidence Threshold | macAUC | HC-macAUC |
|----------------------|--------|-----------|
| T100% (all samples) | 0.732 | — |
| T30% (high confidence) | 0.811 | 0.787 |
| T10% (top confidence) | 0.865 | — |

> Overall suspicious vs. non-suspicious accuracy: **95.08%**

### Feature Importance (Top Features by Gradient Boosting)

```
radius_worst      ████████████████████ 0.481
perimeter_worst   ███████████████████  0.305
area_worst        ███████              0.013
smoothness_worst  ████████████         0.116
```

---

## 🛠️ Tech Stack

```python
# Core ML & Vision
torch / torchvision    # Deep learning (VGG-16, CNN)
scikit-learn           # SVM, Random Forest, Gradient Boosting
xgboost                # Ensemble boosting
opencv-python          # Image processing, active contours
matplotlib             # GUI for cell boundary annotation

# Data Processing
numpy / pandas         # Data manipulation
scipy                  # Statistical tests (KS test)
imbalanced-learn       # SMOTE oversampling
pycaret                # AutoML pipeline for shallow models

# Explainability
grad-cam               # Gradient-weighted Class Activation Maps
statsmodels            # Generalized Linear Models (GLM)

# Infrastructure
flask / fastapi        # Backend API
openpyxl               # Excel export of extracted features
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.9
CUDA-capable GPU (recommended for mammogram training)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/breast-cancer-ai-diagnosis.git
cd breast-cancer-ai-diagnosis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the FNAC Module

```bash
# Launch the Cell Boundary Detector GUI
python fnac/cell_boundary_detector.py --image path/to/fnac_slide.png

# Steps inside the GUI:
# 1. Zoom to a cell of interest
# 2. Click "Draw Manual Boundary" → trace the cell boundary
# 3. Click "Add Snake" to refine with active contour
# 4. Click "Finalize" to extract features and predict malignancy
```

### Running Mammogram Analysis

```bash
# Predict BI-RADS category for a mammogram set
python mammogram/predict.py \
  --r_cc path/to/R_CC.dcm \
  --l_cc path/to/L_CC.dcm \
  --r_mlo path/to/R_MLO.dcm \
  --l_mlo path/to/L_MLO.dcm

# Outputs: BI-RADS score + Grad-CAM heatmap
```

### Running the Full Web App

```bash
# Start the diagnostic platform
python app.py

# Navigate to http://localhost:5000
# Upload patient details, mammograms, or FNAC CSV
# View BI-RADS ratings, malignancy predictions, and Gail risk scores
```

---

## 📁 Dataset Information

| Dataset | Size | Description | Access |
|---------|------|-------------|--------|
| **INBreast** | 115 patients | High-resolution mammograms with BI-RADS annotations | [Request Access](https://www.inbreast.org) |
| **WBCD** (Wisconsin) | 569 records | Nuclear features from FNAC digitized images | [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) |
| **RIMS-FNAC** | 462 records | Custom dataset from real Indian patient data at RIMS, Raichur | Institutional — contact authors |
| **NYU Breast Screening** | 1M+ images | End-to-end CNN pretraining | [NYU Dataset](https://cs.nyu.edu/~kgeras/reports/MRI_datav1.0.pdf) |

> ⚠️ All patient data was collected under appropriate ethical approvals, anonymized, and access-controlled in compliance with data protection standards.

---

## 🔬 Explainability (Grad-CAM)

This system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visually explain model decisions — a critical requirement for clinical adoption.

```
Input Mammogram → CNN Forward Pass → Final Conv Layer Gradients
        → Weighted Feature Map → Heat Overlay on Original Image
```

- ✅ Highlights **microcalcifications** and **masses** — key malignancy indicators
- ✅ Focuses on the **upper outer breast** in MLO views, where most tumors develop
- ✅ Flags clinically irrelevant activations as model quality signals

Additionally, **Generalized Linear Models (GLM)** provide coefficient-level interpretability for FNAC feature contributions.

---

## 🔒 Ethics & Privacy

This project was developed under strict ethical guidelines:

- 📋 **IRB / Ethics Board Approvals** — All patient data collected with institutional approval from RIMS
- 🔐 **HIPAA & GDPR Compliance** — Patient data anonymized and access-controlled
- 🧹 **Minimal Data Collection** — Only necessary patient information was gathered
- 🔏 **Encryption** — Applied to all stored patient data
- 🤝 **Human-in-the-Loop** — AI assists clinicians; final decisions remain with doctors
- 🌐 **Federated Learning (Roadmap)** — Future training will keep data local to institutions

---

## 🔭 Future Scope

```
🧬  GAN-based Synthetic Data       →  Address annotated data scarcity
🌐  Federated Learning             →  Multi-institutional training with privacy
📊  LIME & SHAP Integration        →  Deeper feature-level explainability
🇮🇳  Indian Population Validation   →  Localized model refinement for Indian demographics
🔄  Continuous Online Learning     →  Prevent model drift over time
🏥  COIMBRA Model Integration      →  Advanced risk scoring with genetic factors
```

---


## 📚 References

<details>
<summary>Click to expand full reference list</summary>

1. Tabar L, et al. *The incidence of fatal breast cancer measures the increased effectiveness of therapy in women participating in mammography screening.* Cancer, 2019.
2. Shen, L., et al. *Deep Learning to Improve Breast Cancer Detection on Screening Mammography.* Scientific Reports, 2019.
3. McKinney, S.M., et al. *International evaluation of an AI system for breast cancer screening.* Nature, 2020.
4. Kang, S.H., & Kim, H.J. *Limitations of fine needle aspiration cytology in breast cancer diagnosis.* Journal of Pathology and Translational Medicine, 2017.
5. American College of Radiology. *BI-RADS Atlas, 5th edition.* 2013.
6. Street, W.N., Wolberg, W.H., & Mangasarian, O.L. *Nuclear feature extraction for breast tumor diagnosis.* SPIE Proceedings, 1993.
7. Selvaraju, R.R., et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
8. Gail, M.H., et al. *Projecting Individualized Probabilities of Breast Cancer.* JNCI, 1989.
9. Zamir, R., et al. *Segmenting microcalcifications in mammograms and its applications.* SPIE Medical Imaging, 2021.
10. Kalita, Manjula et al. *A new deep learning model for FNAC image-based breast cancer detection.* IJEECS, 2024.

</details>

---

<div align="center">

**⭐ Star this repository if you find it useful!**

*Made with ❤️ at IIIT Raichur × Bosch Global Software Technologies × RIMS*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
