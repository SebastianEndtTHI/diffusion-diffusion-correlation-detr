# Detection Transformer for Direction-Aware Diffusion-Diffusion Correlation MRI

Conventional MR imaging is limited by the spatial resolution defined by the voxel size of the scan. This means that with standard acquisition techniques, tissue structure can only be determined up to a certain level of accuracy. While a voxel in an MR scan outputs only a single measurement value, this value is actually composed of the sum of signals from the different compartments contained within it. Correlation imaging attempts to reconstruct these partial signals as an inverse problem and thus quantify the microstructure of the tissue. By using diffusion-dependent parameters, directional differences between compartments can also be identified, which can help to determine the course of nerve fibers more accurately.

In this work, we show how a deep learning approach can be used to reconstruct these compartments on the diffusion-imaging level in the form of correlation spectra. Using a model based on a Detection Transformer [1], possible compartment sets for a voxel are predicted and evaluated by an additional existence predictor integrated into the model. We demonstrate that it is fundamentally possible to reconstruct an arbitrary number of compartments within a voxel using the same model, without having to specify the number in advance. With the proposed pipeline, it is also possible to adapt the architecture easily to different measurement protocols without modifying the underlying model structure.

<p align="center">
  <img src="./figures/architecture.svg" alt="Architecture diagram" width="700" />
</p>

This repository provides a full deep-learning pipeline for predicting 
diffusion MRI (dMRI) microstructure compartments using a DETR-inspired 
transformer architecture.  
It includes:

- a transformer-based model (`DWI_DETR`)
- a Hungarian-matching loss for multi-compartment regression
- a complete training and evaluation script
- utilities for logging, data loading, and warm-up scheduling

This project is structured for clarity and reproducibility, enabling researchers 
to train, evaluate, and extend transformer architectures for microstructural 
MRI tasks.

---

## 🚀 Features

- **Transformer-based architecture** for multi-query compartment detection  
- **Hungarian matching loss** computing costs of  
  - Mean diffusivity (MD)  
  - Fractional anisotropy (FA)  
  - Direction vectors (x, y, z)  
  - Compartment weights  
  - Existence query score  
- **Flexible configuration** via `argparse`  
- **Support for pretrained encoders**  
- **Automatic checkpointing + logging**  
- **GPU-ready training loop with warm-up and auxiliary loss options**

---

## 📁 Project Structure

```bash
│
├── datagen/
│   └── dtigenerator.py       # Signal & ground truth generation pipeline
│   └── main.py               # Dataset generator
│
├── dataset/
│   └── dataset_1k_c2_big.csv # Examples for generated datasets
│
├── training/
│   └── main.py               # Training & evaluation pipeline
│   └── dl_models.py          # Transformer model definition
│   └── match_loss.py         # Hungarian matching loss implementation
│   └── train_utils.py        # Training utilities (loader, scheduler, etc.)
│
└── README.md                 # (this file)
```

---

## 📘 Tutorial: How to Train the Model

Below is a step-by-step guide to running a full training session.

--- 

### 1️⃣ Prepare your dataset

Training and test data must be stored as NumPy files or tensors in the format
expected by Train_Utils.get_data():

```markdown
dataset/
    dataset_1k_c2_big.csv
    dataset_1k_c3_big.csv
    dataset_1k_c2_small.csv
    ...
```

Each sample must include:
- input features of shape (input_dim,)
- corresponding target compartments (flattened)

---

### 2️⃣ Run the training script

The default training pipeline can be launched as:

```bash
python main.py \
    --train_data_path /path/to/train_data/ \
    --test_data_path /path/to/test_data/ \
    --model_save_path models/model.pt \
    --log_save_path logs/training_logs.npy
```

---

### 3️⃣ Changing model architecture

You can modify:
- number of transformer decoder layers
- number of decoder queries
- query size / feature space
- hidden dimensions
- multi-head configuration

Example:
```bash
python main.py \
    --n_queries 10 \
    --hidden_dim 512 \
    --n_dlayers 4 \
    --n_multihead 4
```

--- 

### 4️⃣ Training hyperparameters

Example with custom optimizer & scheduler settings:
```bash
python main.py \
    --lr 1e-4 \
    --w_decay 1e-4 \
    --epochs 500 \
    --b_size 256
```

---

### 5️⃣ Loss weights

The Hungarian loss allows flexible weighting of sub-tasks:
```bash
python main.py \
    --md_loss_weight 5.0 \
    --fa_loss_weight 2.0 \
    --dir_loss_weight 0.5 \
    --wt_loss_weight 1.0 \
    --no_obj_weight 0.01
```

---

## 🧠 Model Overview

The architecture consists of:
- Input encoder
- Transformer with multi-head attention
- Learned queries for compartment prediction

Output layers producing:
- MD
- FA
- direction vector (x, y, z)
- compartment weight
- existence score

The model supports auxiliary decoder outputs (`--aux_loss`)
and optional encoder freezing (`--freeze_encoder`).

---

## 📉 Logging and Evaluation

Loss curves and detailed metrics can be extracted from log dictionary with this keys:
```nginx
train_loss
test_loss
test_loss_md
test_loss_fa
test_loss_di
test_loss_wt
test_loss_extnc
```


## References
[1] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-End Object Detection with Transformers,” Lecture Notes in Computer Science, 2020. doi: 10.1007/978-3-030-58452-8_13.
