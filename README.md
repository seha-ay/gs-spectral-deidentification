# GS Spectral De-Identification ReID Pipelines

This directory contains the experiment pipelines used to evaluate Gerchberg-Saxton (GS) spectral transformation for empirical medical-image de-identification, subject re-identification risk reduction, diagnostic utility retention, and reconstruction-based inversion analysis.

The experiments correspond to the study:

**Empirical Characterization of Re-Identification Risk and Diagnostic Utility Under Gerchberg-Saxton Spectral Transformation in Medical Imaging**

The code in this folder supports experiments on:

- **OASIS-1 structural brain MRI**
- **NIH ChestX-ray frontal radiographs**
- **Diagnostic utility evaluation**
- **Cross-domain supervised subject re-identification**
- **In-domain supervised subject re-identification**
- **Embedding-based subject linkage**
- **U-Net reconstruction attacks**
- **Diffusion Posterior Sampling reconstruction attacks**

The repository evaluates empirical re-identification risk under the tested adversary settings. It does **not** claim formal privacy guarantees such as differential privacy.

---

## 1. Repository Scope

This folder is the main experimental pipeline directory:

```text
reID_pipelines/
├── OASIS/
├── NIH-CXR/
└── README.md
```

The two main dataset-specific experiment folders are:

```text
reID_pipelines/OASIS/
reID_pipelines/NIH-CXR/
```

The OASIS folder also contains a patched Diffusion Posterior Sampling module:

```text
reID_pipelines/OASIS/DPS/
```

The DPS module has its own README because it depends on the original external DPS repository and should be treated as a patched reconstruction-adversary component rather than a standalone replacement for the original DPS project.

---

## 2. Scientific Overview

The GS transformation is applied as an input-level preprocessing operation before downstream model training or adversarial evaluation.

At a high level, the modified GS procedure:

1. Starts from a preprocessed grayscale medical image.
2. Applies stochastic phase initialization.
3. Iteratively alternates between spatial and Fourier domains.
4. Normalizes intermediate Fourier magnitude.
5. Applies stochastic frequency-domain masking.
6. Updates the spatial-domain phase.
7. Releases the magnitude of the final inverse-transformed representation.

The transformation is evaluated at masking levels:

```text
RAW
GS-0%
GS-10%
GS-20%
GS-30%
GS-40%
GS-50%
```

GS-0% corresponds to phase perturbation without frequency masking. Higher GS levels introduce progressively stronger stochastic frequency masking.

The main experiments use an iteration count of approximately 50, selected empirically based on convergence, stochastic stability, and computational practicality.

---

## 3. Important Interpretation

This repository evaluates GS as an empirical de-identification-oriented image transformation.

The following interpretation should be maintained:

- GS is a randomized image-domain preprocessing transformation.
- GS does not require labels, subject identifiers, segmentation masks, or task-specific model information.
- GS is applied independently per image.
- GS is compatible with standard downstream classification pipelines.
- GS reduces measured subject-identification risk under the evaluated adversary classes.
- GS does not provide formal privacy guarantees.
- GS should not be described as satisfying differential privacy.
- Masking probability is an algorithmic transformation-strength parameter, not a formal privacy budget.

The privacy-relevant conclusions come from empirical evaluation under defined adversary settings, not from the algorithmic structure alone.

---

## 4. Experimental Design

The full experiment evaluates four main questions.

### 4.1 Diagnostic Utility

Diagnostic utility measures whether task-relevant signal is retained after GS transformation.

Tasks:

```text
OASIS-1  : binary dementia classification
NIH-CXR  : multi-label thoracic pathology classification
```

Models:

```text
ResNet-18
DenseNet-121
```

Utility models are trained and evaluated under matched-domain conditions:

```text
RAW    -> RAW
GS-0%  -> GS-0%
GS-10% -> GS-10%
...
GS-50% -> GS-50%
```

RAW and GS-transformed images are not mixed within a training run.

---

### 4.2 Cross-Domain Supervised Re-Identification

This experiment evaluates whether identity features learned from GS-transformed images transfer back to RAW untransformed images.

Training and testing setup:

```text
Train: GS-transformed images
Test : RAW images
```

This is the primary GS-to-RAW linkage question.

It measures whether a classifier trained on transformed-domain identity structure can identify the same subject in the original RAW-image domain.

---

### 4.3 In-Domain Supervised Re-Identification

This experiment evaluates residual identity information that remains learnable entirely inside the transformed GS domain.

Training and testing setup:

```text
Train: GS-transformed images
Test : GS-transformed images at the same masking level
```

This is a matched-domain closed-set subject-identification experiment.

It should be interpreted as an upper-bound style evaluation of identity learnability within the transformed representation, not as direct RAW identity recovery.

---

### 4.4 Embedding-Based Re-Identification

This experiment evaluates subject linkage using embeddings from utility-trained models.

Embeddings are extracted from:

```text
ResNet-18    : average pooling layer, 512 dimensions
DenseNet-121 : feature pooling layer, 1024 dimensions
```

Embeddings are L2-normalized and matched using cosine similarity.

This is an open-set subject-level linkage evaluation using held-out subjects.

Gallery sizes include:

```text
k = 1
k = 3
k = 5
```

Some supplemental analyses may include larger gallery settings, depending on the dataset and notebook.

---

### 4.5 Reconstruction-Based Inversion

This experiment evaluates whether RAW images can be reconstructed from GS-transformed images and whether those reconstructions preserve identity-bearing structure.

Two reconstruction paradigms are evaluated:

```text
U-Net reconstruction
Diffusion Posterior Sampling reconstruction
```

Reconstruction experiments focus on:

```text
GS-0%
GS-50%
```

The reconstruction outputs are evaluated using:

```text
SSIM
PSNR
subject re-identification accuracy from reconstructed images
```

The key distinction is that perceptual reconstruction fidelity is not treated as equivalent to subject identity recovery.

---

## 5. Dataset Summary

### 5.1 OASIS-1

OASIS-1 is used for structural brain MRI experiments.

Primary task:

```text
binary dementia classification
```

Main evaluation types:

```text
diagnostic utility
cross-domain supervised ReID
in-domain supervised ReID
embedding-based ReID
U-Net reconstruction
DPS reconstruction
```

The OASIS experiments use subject-level splits for utility and embedding-based ReID, and slice-level controlled splits for supervised ReID and reconstruction-based inversion.

---

### 5.2 NIH ChestX-ray

NIH-CXR is used for frontal chest radiograph experiments.

Primary task:

```text
11-class multi-label thoracic pathology classification
```

Main evaluation types:

```text
diagnostic utility
cross-domain supervised ReID
in-domain supervised ReID
embedding-based ReID
U-Net reconstruction
```

The NIH-CXR experiments use subject-level splits for utility and embedding-based ReID, and image-level controlled splits for supervised ReID.

---

## 6. Directory Structure

Current high-level structure:

```text
reID_pipelines/
├── OASIS/
│   ├── 01_data_prep.ipynb
│   ├── 02_OASIS_utility.ipynb
│   ├── 03_OASIS_reid_PIDs.ipynb
│   ├── 04_OASIS_reid_featureExt.ipynb
│   ├── 05_OASIS_UNet.ipynb
│   ├── OASIS_Recon_CIs.ipynb
│   ├── gs_functions.py
│   ├── DPS/
│   └── results/
│
├── NIH-CXR/
│   ├── data_prep_download.ipynb
│   ├── 01-utility.ipynb
│   ├── 02-reid-PID.ipynb
│   ├── 03-UNet.ipynb
│   ├── 04-reID_Feature.ipynb
│   ├── NIH-get-CIs.ipynb
│   ├── gs_functions.py
│   └── results/
│
└── README.md
```

---

## 7. Notebook Execution Order

The notebooks are organized by dataset and experiment type.

### 7.1 OASIS Pipeline

Recommended order:

```text
01_data_prep.ipynb
02_OASIS_utility.ipynb
03_OASIS_reid_PIDs.ipynb
04_OASIS_reid_featureExt.ipynb
05_OASIS_UNet.ipynb
OASIS_Recon_CIs.ipynb
```

Purpose of each notebook:

```text
01_data_prep.ipynb
    Prepares OASIS image data and experiment splits.

02_OASIS_utility.ipynb
    Trains and evaluates diagnostic utility models across RAW and GS levels.

03_OASIS_reid_PIDs.ipynb
    Runs supervised subject re-identification experiments.

04_OASIS_reid_featureExt.ipynb
    Runs embedding-based subject linkage using utility-model embeddings.

05_OASIS_UNet.ipynb
    Runs U-Net reconstruction experiments.

OASIS_Recon_CIs.ipynb
    Computes confidence intervals and summaries for reconstruction outputs.
```

The DPS reconstruction experiments are contained separately under:

```text
OASIS/DPS/
```

See:

```text
OASIS/DPS/README.md
```

for the DPS-specific setup and patch instructions.

---

### 7.2 NIH-CXR Pipeline

Recommended order:

```text
data_prep_download.ipynb
01-utility.ipynb
02-reid-PID.ipynb
04-reID_Feature.ipynb
03-UNet.ipynb
NIH-get-CIs.ipynb
```

Purpose of each notebook:

```text
data_prep_download.ipynb
    Prepares or downloads NIH-CXR data and organizes experiment inputs.

01-utility.ipynb
    Trains and evaluates multi-label diagnostic utility models.

02-reid-PID.ipynb
    Runs supervised subject re-identification experiments.

04-reID_Feature.ipynb
    Runs embedding-based subject linkage experiments.

03-UNet.ipynb
    Runs U-Net reconstruction experiments.

NIH-get-CIs.ipynb
    Computes confidence intervals and final result summaries.
```

---

## 8. GS Function Files

Both dataset folders include a GS helper file:

```text
OASIS/gs_functions.py
NIH-CXR/gs_functions.py
```

These files contain dataset-local GS transformation utilities used by the notebooks.

The transformation should be applied after standard preprocessing and before model training or evaluation.

Typical preprocessing assumptions:

```text
grayscale image format
image resized to 224 x 224
intensity normalized to 0-1
GS applied independently per image
```

---

## 9. Results Included in This Directory

This repository includes selected result summaries and representative outputs.

### 9.1 OASIS Results

Main OASIS result folders:

```text
OASIS/results/oasis/utility/
OASIS/results/oasis/reid/
OASIS/results/oasis/reid_feature_ext/
OASIS/results/oasis/recon/
```

Important summary files include:

```text
OASIS/results/oasis/utility/per_fold_results.csv
OASIS/results/oasis/utility/test_results.csv

OASIS/results/oasis/reid/OASIS_Complete_Results.csv
OASIS/results/oasis/reid/OASIS_resnet18_Pretrained_results.csv
OASIS/results/oasis/reid/OASIS_resnet18_Scratch_results.csv
OASIS/results/oasis/reid/OASIS_densenet121_Pretrained_results.csv
OASIS/results/oasis/reid/OASIS_densenet121_Scratch_results.csv

OASIS/results/oasis/reid_feature_ext/OASIS_ReID_FeatureExt.csv
OASIS/results/oasis/reid_feature_ext/OASIS_ReID_FeatureExt_K61.csv

OASIS/results/oasis/recon/oasis_unet_recon_ci.csv
OASIS/results/oasis/recon/oasis_dps_reid_ci.csv
OASIS/results/oasis/recon/oasis_dps_intensity_ci.csv
```

Some `.pkl` files under OASIS results are intermediate visualization or saliency artifacts. They are useful for reproducing specific plots, but they are not required for understanding the main numerical summaries.

---

### 9.2 NIH-CXR Results

Main NIH-CXR result folders:

```text
NIH-CXR/results/nih_cxr/utility/
NIH-CXR/results/nih_cxr/reid/
NIH-CXR/results/nih_cxr/reid_feature_ext/
NIH-CXR/results/nih_cxr/unet/
```

Important summary files include:

```text
NIH-CXR/results/nih_cxr/utility/per_run_results.csv
NIH-CXR/results/nih_cxr/reid/NIH_CXR_ReID_Results.csv
NIH-CXR/results/nih_cxr/reid_feature_ext/NIH_CXR_ReID_FeatureExt.csv
NIH-CXR/results/nih_cxr/unet/functionality_results.csv
NIH-CXR/results/nih_cxr/unet/intensity_results.csv
```

---

## 10. Diffusion Posterior Sampling Component

The DPS reconstruction component is located at:

```text
OASIS/DPS/
```

This component adapts the original Diffusion Posterior Sampling repository for OASIS GS reconstruction experiments.

Original DPS repository:

```text
https://github.com/DPS2022/diffusion-posterior-sampling/tree/main
```

The DPS folder in this repository is a patch-oriented module. It should be used together with the original DPS repository.

See the DPS-specific README:

```text
OASIS/DPS/README.md
```

for instructions on how to apply the patch, obtain the original `ffhq_10m.pt` checkpoint, and reproduce DPS reconstruction experiments.

Important model files:

```text
OASIS/DPS/models/ffhq_10m.pt
OASIS/DPS/checkpoints/brain_finetune/best_model.pt
```

The `ffhq_10m.pt` model is the original FFHQ-pretrained DPS checkpoint used as the base model for brain-MRI fine-tuning. It is not necessarily included in this repository and should be obtained from the public resources associated with the original DPS project.

The `best_model.pt` file is the fine-tuned brain-MRI checkpoint used for the OASIS DPS reconstruction experiments. Depending on repository size constraints, this file may be excluded from the GitHub release and distributed separately.

Expected location if available:

```text
OASIS/DPS/checkpoints/brain_finetune/best_model.pt
```

---

## 11. Files Intentionally Excluded or Not Recommended for Git Tracking

Large model files, checkpoints, generated caches, and notebook checkpoint folders should generally not be tracked in normal Git.

Recommended exclusions:

```text
**/.ipynb_checkpoints/
**/__pycache__/
*.pyc

*.pt
*.pth
*.ckpt
*.safetensors
*.onnx

*.npy
*.npz
*.pkl
*.pickle
*.h5
*.hdf5

*.zip
*.tar
*.tar.gz
*.tgz
*.7z
```

The largest known local files include:

```text
OASIS/DPS/checkpoints/brain_finetune/best_model.pt
OASIS/DPS/checkpoints/brain_finetune/opt000000.pt
```

These files may exceed normal GitHub file-size limits and should be handled using one of the following:

```text
external download instructions
institutional storage link
GitHub Releases
Git LFS, if appropriate
manual placement after cloning
```

---

## 12. Data Availability

The datasets used in these experiments are publicly available medical imaging datasets.

Datasets are not included in this repository.

Users must obtain them separately from their official sources and comply with each dataset's access terms, license, and citation requirements.

Datasets:

```text
OASIS-1 structural MRI
NIH ChestX-ray
```

After downloading, users should update local paths inside the data-preparation notebooks and any path variables used by the training/evaluation notebooks.

---

## 13. Reproduction Workflow

A typical reproduction workflow is:

```text
1. Clone this repository.

2. Download the required public datasets:
   - OASIS-1
   - NIH ChestX-ray

3. Update dataset paths inside the data-preparation notebooks.

4. Run OASIS data preparation:
   OASIS/01_data_prep.ipynb

5. Run NIH-CXR data preparation:
   NIH-CXR/data_prep_download.ipynb

6. Run diagnostic utility notebooks.

7. Run supervised ReID notebooks.

8. Run embedding-based ReID notebooks.

9. Run U-Net reconstruction notebooks.

10. For OASIS DPS reconstruction:
    follow OASIS/DPS/README.md.

11. Run confidence-interval notebooks.

12. Compare output CSV summaries against the included result files.
```

---

## 14. Expected GS Levels

Unless otherwise noted, experiments use the following transformation conditions:

```text
RAW
GS-0%
GS-10%
GS-20%
GS-30%
GS-40%
GS-50%
```

Utility and supervised/embedding ReID experiments generally evaluate the full set of GS levels.

Reconstruction experiments focus primarily on:

```text
GS-0%
GS-50%
```

These represent the weakest and strongest GS reconstruction conditions evaluated in the main reconstruction-adversary analysis.

---

## 15. Evaluation Metrics

### 15.1 Diagnostic Utility

Diagnostic utility is evaluated using:

```text
AUC
macro-AUC where applicable
macro-F1 where applicable
subject-level aggregation
confidence intervals
```

### 15.2 Supervised Re-Identification

Supervised ReID is evaluated using:

```text
Top-1 subject-identification accuracy
95% confidence intervals
```

Cross-domain ReID:

```text
Train on GS
Test on RAW
```

In-domain ReID:

```text
Train on GS
Test on GS
```

### 15.3 Embedding-Based Re-Identification

Embedding-based ReID is evaluated using:

```text
Rank-1 identification accuracy
cosine similarity
L2-normalized penultimate-layer embeddings
gallery sizes k = 1, 3, 5
```

### 15.4 Reconstruction

Reconstruction outputs are evaluated using:

```text
SSIM
PSNR
ReID accuracy from reconstructed images
```

The reconstruction experiments are designed to test whether visually plausible recovery also restores subject-identifying structure.

---

## 16. Environment

The notebooks were developed in a Jupyter environment with GPU support.

Core Python dependencies include:

```text
python
jupyter
numpy
pandas
scipy
scikit-learn
matplotlib
pillow
opencv-python
torch
torchvision
tqdm
```

Additional dependencies may be required for specific notebooks or DPS reconstruction.

For the DPS component, consult:

```text
OASIS/DPS/README.md
```

and the original DPS repository instructions.

---

## 17. Reproducibility Notes

Several experiments involve stochastic components:

```text
random GS phase initialization
per-iteration stochastic frequency masks
neural network initialization
train/validation/test splitting
data loader ordering
GPU nondeterminism
```

For exact reproducibility, users should:

```text
set random seeds
preserve fixed split files
avoid regenerating GS-transformed datasets unless intended
record package versions
record CUDA/PyTorch versions
preserve model checkpoints when comparing trained models
```

In the reported experiments, transformed datasets were generated once per condition and reused across downstream evaluations so that observed differences reflected the transformation condition rather than repeated stochastic resampling.

---

## 18. GitHub Upload Notes

This folder may contain files that are too large for normal GitHub tracking.

Before pushing, check:

```bash
du -sh reID_pipelines

find reID_pipelines -type f -size +100M -exec ls -lh {} \;

find reID_pipelines -type f \( \
    -name "*.pt" -o \
    -name "*.pth" -o \
    -name "*.ckpt" -o \
    -name "*.pkl" -o \
    -name "*.npy" -o \
    -name "*.npz" \
\) -exec ls -lh {} \;
```

Do not push large checkpoint files through normal Git.

Recommended approach:

```text
Track:
    notebooks
    Python scripts
    configuration files
    small CSV result summaries
    README files
    selected representative images

Do not track:
    full model checkpoints
    large training outputs
    full generated reconstruction folders
    .ipynb_checkpoints
    large intermediate pickle caches
```

---

## 19. Citation

If using this repository, cite the corresponding manuscript:

```text
Ay, Seha, Wei Zhang, and Umit Topaloglu.
Empirical Characterization of Re-Identification Risk and Diagnostic Utility
Under Gerchberg-Saxton Spectral Transformation in Medical Imaging.
```

If using the DPS reconstruction component, also cite the original DPS work:

```bibtex
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```

Users should also cite the original dataset sources for OASIS-1 and NIH ChestX-ray.

---

## 20. Contact

For questions about this experimental pipeline, contact the repository maintainer:

```text
Seha Ay
Wake Forest School of Medicine
Department of Biomedical Engineering
seha.ay@wfusm.edu
```