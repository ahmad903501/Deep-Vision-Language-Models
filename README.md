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
