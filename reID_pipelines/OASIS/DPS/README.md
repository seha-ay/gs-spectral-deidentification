# DPS Patch for OASIS GS Reconstruction Experiments

This directory contains the experiment-specific patch files used to adapt the original **Diffusion Posterior Sampling** repository for OASIS brain-MRI reconstruction experiments under Gerchberg-Saxton (GS) transformation.

This is **not** a standalone redistribution of the original DPS repository. To reproduce the experiments, users must first obtain the original DPS codebase from the official repository and then apply the files in this patch directory.

Main DPS repository:

https://github.com/DPS2022/diffusion-posterior-sampling/tree/main

Original DPS paper/repository:

**Diffusion Posterior Sampling for General Noisy Inverse Problems**  
Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, Jong Chul Ye  
ICLR 2023 Spotlight

---

## 1. Purpose of This Patch

The original DPS repository provides a general framework for solving noisy inverse problems using diffusion posterior sampling. This patch adapts that framework for the reconstruction component of our OASIS brain-MRI GS experiments.

The patched version supports reconstruction experiments involving:

1. OASIS brain-MRI inputs.
2. GS-transformed OASIS images.
3. A modified dataloader and sampling pipeline.
4. A task-specific GS inversion configuration.
5. Brain-MRI fine-tuning scripts.
6. Example reconstruction outputs from both:
   - the original FFHQ-pretrained model without brain-MRI fine-tuning, and
   - the fine-tuned brain-MRI reconstruction model.

The goal is to allow users to reconstruct the DPS experiment environment without redistributing the full original DPS codebase as a modified fork.

---

## 2. Required Base Repository

Clone the official DPS repository first:

```bash
git clone https://github.com/DPS2022/diffusion-posterior-sampling
cd diffusion-posterior-sampling
```

## 3.FFHQ Base Model

The original FFHQ-pretrained DPS model checkpoint, `ffhq_10m.pt`, is not included in this repository due to file-size and redistribution considerations.

This checkpoint was used as the base model for brain-MRI fine-tuning. To reproduce the fine-tuning workflow, users should obtain `ffhq_10m.pt` from the public resources associated with the original DPS repository and place it at:

```text
models/ffhq_10m.pt
```

## 4. Expected Directory Structure

diffusion-posterior-sampling/
├── config/
│   └── gs_inversion_config.yaml
│
├── data/
│   └── dataloader.py
│
├── guided_diffusion/
│   ├── dist_util.py
│   ├── image_dataset.py
│   ├── logger.py
│   ├── losses.py
│   ├── resample.py
│   ├── respace.py
│   ├── script_util.py
│   └── train_util.py
│
├── models/
│   └── ffhq_10m.pt
│
├── checkpoints/
│   └── [fine-tuned brain-MRI model checkpoint]
│
├── results/
│   ├── no_fine_tuning/
│   └── finetuned/
│       ├── OASIS_bench_finetuned/
│       │   ├── noise/
│       │   │   ├── input/00000.png
│       │   │   ├── label/00000.png
│       │   │   ├── progress/00000.png
│       │   │   └── recon/00000.png
│       │   └── OASIS_bench_finetuned.log
│       │
│       ├── OASIS_gs0p_finetuned/
│       │   ├── noise/
│       │   │   ├── input/00000.png
│       │   │   ├── label/00000.png
│       │   │   ├── progress/00000.png
│       │   │   └── recon/00000.png
│       │   └── OASIS_gs0p_finetuned.log
│       │
│       └── OASIS_gs50p_finetuned/
│           ├── noise/
│           │   ├── input/00000.png
│           │   ├── label/00000.png
│           │   ├── progress/00000.png
│           │   └── recon/00000.png
│           └── OASIS_gs50p_finetuned.log
│
├── scripts/
│   ├── finetune_brain.py
│   └── image_train.py
│
└── sample_condition.py