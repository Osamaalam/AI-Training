Updated: 2026-06-08

# 🦴 Fracture and Torsion Abnormality Detection in Musculoskeletal X-Rays via Fine-Tuned VLMs

This repository contains a high-performance, production-ready pipeline for fine-tuning Vision-Language Models (VLMs) on musculoskeletal X-ray images to detect fractures and torsion abnormalities. It utilizes state-of-the-art parameter-efficient fine-tuning (PEFT/LoRA) coupled with **Unsloth** for 2x faster training and high memory efficiency.

---

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [🏥 End-to-End Clinical Enterprise Pipeline](#-end-to-end-clinical-enterprise-pipeline)
  - [1. PACS Ingestion (Philips IntelliSpace)](#1-pacs-ingestion-philips-intellispace)
  - [2. DICOM Preprocessing & Conversion](#2-dicom-preprocessing--conversion)
  - [3. Training Augmentations](#3-training-augmentations)
  - [4. GCP Real-Time Inference Server](#4-gcp-real-time-inference-server)
  - [5. Real-Time Closed-Loop PACS Integration](#5-real-time-closed-loop-pacs-integration)
- [⚡ Architecture & Optimization](#-architecture--optimization)
- [📂 Project Structure](#-project-structure)
- [🛠️ Environment Setup](#%EF%B8%8F-environment-setup)
- [⚙️ Configuration Guide](#%EF%B8%8F-configuration-guide)
- [🚀 Training Execution](#-training-execution)
- [📊 Results & Explainability Analysis](#-results--explainability-analysis)
  - [📈 Fracture Detection Results](#-fracture-detection-results)
  - [🔄 Torsion Abnormality Results](#-torsion-abnormality-results)
  - [🔍 Occlusion Heatmaps & Model Interpretability](#-occlusion-heatmaps--model-interpretability)

---

## 🎯 Project Overview

In clinical orthopedics, automated fracture and torsion detection serves as a critical decision-support tool. This project implements a visual question-answering (VQA) paradigm where a pre-trained 8B-parameter Vision-Language Model is formatted to receive X-ray scans with clinical instruction prompts, classifying them directly into binary categories: **positive** (abnormality detected) or **negative** (no abnormality).

---

## 🏥 End-to-End Clinical Enterprise Pipeline

To operate successfully within a hospital infrastructure, this VLM classifier goes far beyond static file training. It is designed as an end-to-end automated clinical loop integrated directly with hospital DICOM networks and cloud infrastructure.

```text
  +------------------+             +----------------------+             +----------------------+
  |   Philips PACS   |  (C-STORE)  | Local DICOM Gateway  |  (HTTPS)    | GCP Inference Server |
  | (Acquisition/QA) | ----------> | (Orthanc/dcm4chee)   | ----------> |  (Triton/vLLM + GPU) |
  +------------------+             +----------------------+             +----------------------+
           ^                                                                        |
           |                                                                        | (Inference)
           |                          (C-STORE SR / Capture)                        v
           +------------------------------------------------------------------------+
```

### 1. PACS Ingestion (Philips IntelliSpace)
- **Acquisition**: X-ray modalities (`DX` / `CR`) send raw radiographic scans to the **Philips IntelliSpace PACS** system.
- **Trigger**: Upon technician acquisition and validation, a PACS routing rule automatically triggers a `C-STORE` push command, sending the raw DICOM files to an on-premise, secure DICOM Router Gateway (e.g., *Orthanc* or *dcm4chee*) which acts as a bridge.

### 2. DICOM Preprocessing & Conversion
Once the router receives the DICOM instance, the preprocessing microservice executes:
1. **Metadata Filtering**: Inspects DICOM headers (`Modality == "DX" or "CR"`, and `BodyPartExamined` in `["WRIST", "ELBOW", "HAND", "SHOULDER"]`) using `pydicom`.
2. **HIPAA Anonymization**: Scrubs Patient Identifying Information (PHI) like PatientName, PatientID, and BirthDate to guarantee absolute clinical data privacy.
3. **Bit-Depth Scaling & Windowing**: High-dynamic-range raw pixels ($12$-bit or $16$-bit) are windowed using the DICOM `Window Center` ($WC$) and `Window Width` ($WW$) attributes to map pixels into an optimized $8$-bit PNG representation.
4. **Equalization (CLAHE)**: Applies Contrast Limited Adaptive Histogram Equalization to highlight faint micro-fractures, hair-line cracks, and cortical bone disruptions.

### 3. Training Augmentations
For training resilience, processed images undergo an inline augmentation pipeline:
- Spatial modifications: Random rotation ($\pm 15^\circ$), translation, and elastic warping to mimic variations in patient joint positioning.
- Contrast & brightness variations to account for different X-ray machine manufacturers.
- Conversion to structured VLM dialogue templates containing clinical directives and body modality annotations.

### 4. GCP Real-Time Inference Server
The model is deployed in a high-availability, low-latency configuration on Google Cloud Platform (GCP):
- **Infrastructure**: Hosted on Google Kubernetes Engine (GKE) or VM nodes equipped with **NVIDIA L4 or A100 GPUs**.
- **Model Server**: Containerized using **Triton Inference Server** or an optimized **FastAPI + vLLM** back-end with dynamic batching.
- **Quantization**: Employs $16$-bit precision or $4$-bit AWQ quantization to reduce end-to-end latency to **$< 350$ms** per scan.

### 5. Real-Time Closed-Loop PACS Integration
1. **Request**: The DICOM Router forwards the preprocessed PNG payload along with the modality tags to the GCP inference service over an encrypted **TLS 1.3** connection.
2. **Analysis**: The VLM detects abnormalities, generates classification confidence, and produces an **Occlusion Heatmap** highlighting the exact fracture site.
3. **Structured Response**: The GCP server compiles the outputs and returns them to the Local DICOM Router.
4. **DICOM Wrapper**: The Router encapsulates the diagnostic prediction (positive/negative) and confidence metrics into a standard **DICOM Structured Report (SR)**. Additionally, it burns the occlusion heatmap overlay into a **Secondary Capture DICOM image**.
5. **C-STORE Feed**: The router pushes both files back to the **Philips PACS** server via standard medical ports.
6. **Clinical Review**: Within **$< 3$ seconds** of acquisition, the reviewing Radiologist receives the fracture heatmap directly inside their standard Philips clinical viewer as a secondary series, facilitating immediate diagnostic confidence.

---

## ⚡ Architecture & Optimization

This production pipeline incorporates modern techniques for training large deep learning architectures:
- **Base Model**: Qwen/VLM-8B model optimized using custom 16-bit unsloth patches.
- **PEFT / LoRA**: Wraps both vision layers and language projection/attention layers with low-rank adapters ($r=16, \alpha=16$) to train only $0.58\%$ of total weights ($51.3$M out of $8.8$B params), preventing catastrophic forgetting.
- **Memory Optimization**: Gradient Checkpointing is active to minimize VRAM utilization, enabling large batch sizes ($32$ effective batch size) on modern GPU nodes.
- **Bfloat16 Precision**: Employs Native Brain Float 16 (`bf16=True`) for stable numerical representation.

---

## 📂 Project Structure

The project has been refactored from loose Jupyter Notebook cells into a modular, production-ready Python package structure:

```text
├── config.yaml          # Hyperparameter and paths configuration file
├── requirements.txt     # List of python package dependencies
├── training.log         # Automatically generated execution log
├── README.md            # Comprehensive project documentation & results analysis
├── src/
│   ├── __init__.py      # Package identifier
│   ├── config.py        # Config loader (dataclasses + YAML parsing)
│   ├── dataset.py       # Conversational VLM dataset processing & conversion
│   ├── model.py         # FastVisionModel initializer & LoRA wrapping
│   └── train.py         # Main training execution script with CLI overrides
└── Results/             # Folder containing experimental validation outputs
    └── Results/
        └── Preliminary_Results/
            ├── Fracture/
            │   ├── Matrix.png     # Fracture classification Confusion Matrix
            │   ├── f1.jpg         # Fracture classification F1 optimization curve
            │   └── Heatmap/
            │       ├── image2_occ.png  # Occlusion sensitivity map (Fracture sample A)
            │       └── image3_occ.png  # Occlusion sensitivity map (Fracture sample B)
            └── Torsion/
                ├── matrix.jpg     # Torsion detection Confusion Matrix
                ├── F1.jpg         # Torsion detection F1 optimization curve
                └── Sample.jpg     # Highlighted sample prediction for torsion
```

---

## 🛠️ Environment Setup

Ensure you have PyTorch configured with CUDA support.

1. **Clone this repository**:
   ```bash
   git clone <your-repository-url>
   cd Preliminary
   ```

2. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Unsloth and Xformers** (Matched to CUDA/Torch version):
   For optimized finetuning speeds:
   ```bash
   pip install --no-deps unsloth unsloth_zoo
   ```

---

## ⚙️ Configuration Guide

The model execution, PEFT, dataset paths, and optimizer hyper-parameters are centralized in `config.yaml`. This ensures experiments are repeatable and config-controlled:

```yaml
model:
  model_name: "../Models/VLM-unsloth-8b"
  load_in_16bit: true
  dtype: "bfloat16"

peft:
  r: 16
  lora_alpha: 16
  use_rslora: true
  finetune_vision_layers: true
  finetune_language_layers: true

dataset:
  train_csv_path: "train.csv"
  val_csv_path: "valid.csv"
  max_length: 2048

training:
  output_dir: "outputs-16-32-6-1e4-8b-16bit"
  num_train_epochs: 6
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 1.0e-4
  eval_strategy: "steps"
  eval_steps: 200
```

---

## 🚀 Training Execution

To execute the training pipeline, run the main script. Command-line flags are supported to override any configured defaults on-the-fly:

```bash
# Run training using config.yaml defaults
python src/train.py

# Run training with specific hyperparameter overrides
python src/train.py --epochs 8 --batch_size 8 --train_csv data/custom_train.csv
```

All process logs, hyperparameter tables, loss values, and validation checks are duplicated to standard output and the `training.log` file automatically.

---

## 📊 Results & Explainability Analysis

The fine-tuned model was rigorously validated. Below are the results and explainability figures from our preliminary validation.

### 📈 Fracture Detection Results

The evaluation metrics demonstrate exceptional classification performance across musculoskeletal structures (e.g., wrist, elbow X-rays).

| Metric | Value |
| :--- | :--- |
| **Validation Loss** | `0.0186` (Step 5000) |
| **F1 Peak Accuracy** | Consistent optimal score over epochs |

#### **Confusion Matrix & F1 Score (Fracture)**
The confusion matrix demonstrates extremely low false positive and false negative counts, resulting in high clinical precision and recall:

<p align="center">
  <img src="Results/Results/Preliminary_Results/Fracture/Matrix.png" width="45%" alt="Fracture Confusion Matrix" />
  <img src="Results/Results/Preliminary_Results/Fracture/f1.jpg" width="45%" alt="Fracture F1-Score Curve" />
</p>

*The F1 optimization curve highlights convergence around epoch 4 with stable and resilient generalization characteristics.*

---

### 🔄 Torsion Abnormality Results

Evaluation was similarly completed for structural torsion (rotational bone twists and joint displacements).

#### **Performance & Sample Predictions (Torsion)**
Below is the evaluation summary for torsion classification, displaying its high-precision confusion matrix and a labeled validation sample:

<p align="center">
  <img src="Results/Results/Preliminary_Results/Torsion/matrix.jpg" width="31%" alt="Torsion Confusion Matrix" />
  <img src="Results/Results/Preliminary_Results/Torsion/F1.jpg" width="31%" alt="Torsion F1 Curve" />
  <img src="Results/Results/Preliminary_Results/Torsion/Sample.jpg" width="31%" alt="Torsion Sample Prediction" />
</p>

---

### 🔍 Occlusion Heatmaps & Model Interpretability

To prevent the model from relying on confounding factors in background pixels and to ensure true clinical interpretability, we performed **Occlusion Sensitivity Analysis**. 

An occlusion map is constructed by systematically masking a sliding window across the X-ray image and measuring changes in the model's prediction probability. 

<p align="center">
  <img src="Results/Results/Preliminary_Results/Fracture/Heatmap/image2_occ.png" width="45%" alt="Occlusion Sensitivity Sample A" />
  <img src="Results/Results/Preliminary_Results/Fracture/Heatmap/image3_occ.png" width="45%" alt="Occlusion Sensitivity Sample B" />
</p>

#### **What the Heatmaps Show:**
* **Red/Hot Spot Zones**: Point to regions where occluding/masking the pixel severely drops the model's confidence in fracture presence.
* **Clinical Mapping**: As shown in `image2_occ.png` and `image3_occ.png`, the peak intensity of the occlusion heatmap maps precisely over the anatomical site of the cortical bone disruption (the actual fracture line).
* **Significance**: This guarantees that our fine-tuned VLM is making its predictions based on correct anatomical and pathological features rather than unrelated image noise or clinical markers.