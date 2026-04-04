# DVLM PA1 DDPM Implementation

This project implements the coding section of DVLM PA1 in a modular, testable way.

## Phases implemented

1. Project architecture and setup
2. Diffusion math and schedules
3. U-Net epsilon predictor with timestep conditioning
4. Training and sanity gates
5. Ancestral sampling and qualitative artifacts
6. Evaluation and controlled experiments

## Quick start

1. Install dependencies:

   pip install -r requirements.txt

2. Run baseline training:

   python train.py --config configs/default.yaml

3. Run evaluation:

   python eval.py --config configs/default.yaml

## Push to GitHub

1. Create a new repository on GitHub (do not initialize with README).
2. Add your remote and push:

   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main

3. Confirm the repository is visible on GitHub.

## Run on Kaggle GPU

### Option A: Public GitHub repo (recommended)

1. Open Kaggle -> Code -> New Notebook.
2. In Notebook settings, set Accelerator to GPU.
3. In the first cell:

   !git clone https://github.com/<your-username>/<your-repo>.git
   %cd <your-repo>
   !pip install -r requirements.txt

4. Train with Kaggle config:

   !python train.py --config configs/kaggle.yaml

5. Evaluate with generated checkpoint:

   !python eval.py --config configs/kaggle.yaml --checkpoint /kaggle/working/artifacts/ddpm_mnist_kaggle/<run_stamp>/checkpoints/final.pt --num-samples 10000

6. Download outputs from /kaggle/working/artifacts.

### Option B: Private repo

Kaggle cannot clone private repos without credentials. Use either:
- A Kaggle Dataset created from a zipped copy of this project, or
- A GitHub personal access token flow in notebook (less convenient and less secure).

### Notes for Kaggle runtime limits

- The baseline config in configs/default.yaml is heavy.
- Use configs/kaggle.yaml first (20k steps) to validate end-to-end.
- Increase max_steps later if your session budget allows.

## Mandatory artifacts

- Real data grid (8x8)
- Alpha-bar and SNR plots
- Training loss curve
- Final sample grid (64 samples)
- Denoising trajectory snapshots
- Metrics report (FID/KID, BPD, nearest-neighbor memorization check)

## Structure

- `src/data`: dataset and preprocessing
- `src/diffusion`: schedule, forward, posterior, and samplers
- `src/models`: U-Net epsilon predictor
- `src/train`: training logic and sanity checks
- `src/eval`: plotting and metrics
- `src/experiments`: ablations and extension runners
- `src/utils`: config, seed, and utility helpers
