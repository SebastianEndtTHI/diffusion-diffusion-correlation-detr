# Detection Transformer for Direction-Aware Diffusion-Diffusion Correlation MRI

Conventional MR imaging is limited by the spatial resolution defined by the voxel size of the scan. This means that with standard acquisition techniques, tissue structure can only be determined up to a certain level of accuracy. While a voxel in an MR scan outputs only a single measurement value, this value is actually composed of the sum of signals from the different compartments contained within it. Correlation imaging attempts to reconstruct these partial signals as an inverse problem and thus quantify the microstructure of the tissue. By using diffusion-dependent parameters, directional differences between compartments can also be identified, which can help to determine the course of nerve fibers more accurately.

In this work, we show how a deep learning approach can be used to reconstruct these compartments on the diffusion-imaging level in the form of correlation spectra. Using a model based on a Detection Transformer [1], possible compartment sets for a voxel are predicted and evaluated by an additional existence predictor integrated into the model. We demonstrate that it is fundamentally possible to reconstruct an arbitrary number of compartments within a voxel using the same model, without having to specify the number in advance. With the proposed pipeline, it is also possible to adapt the architecture easily to different measurement protocols without modifying the underlying model structure.

This repository provides a full deep-learning pipeline for predicting 
diffusion MRI (dMRI) microstructure compartments using a DETR-inspired 
transformer architecture.  
It includes:

- a transformer-based model (`DWI_DETR_Att`)
- a Hungarian-matching loss for multi-compartment regression
- a complete training and evaluation script
- utilities for logging, data loading, and warm-up scheduling

This project is structured for clarity and reproducibility, enabling researchers 
to train, evaluate, and extend transformer architectures for microstructural 
MRI tasks.

---

## 🚀 Features

- **Transformer-based architecture** for multi-query compartment regression  
- **Hungarian matching loss** supporting  
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

# 📦 Installation

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

## References
[1] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-End Object Detection with Transformers,” Lecture Notes in Computer Science, 2020. doi: 10.1007/978-3-030-58452-8_13.
