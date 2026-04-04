from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

from src.diffusion.forward import q_sample, sample_timesteps
from src.diffusion.posterior import p_mean_from_eps, q_posterior_mean_var
from src.diffusion.schedule import ScheduleState


@dataclass
class MetricReport:
    dataset_fid: float
    dataset_kid: float
    bpd: float
    nearest_neighbor_l2: float


class SmallFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, num_classes)
        self.feature_proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        logits = self.head(h)
        feats = self.feature_proj(h)
        return logits, feats


def train_feature_extractor(
    model: SmallFeatureExtractor,
    loader: Iterable,
    device: torch.device,
    epochs: int = 2,
) -> SmallFeatureExtractor:
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def _collect_features(
    extractor: SmallFeatureExtractor,
    loader: Iterable,
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    feats = []
    total = 0
    extractor.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, f = extractor(x)
            feats.append(f.detach().cpu().numpy())
            total += x.size(0)
            if total >= max_samples:
                break
    out = np.concatenate(feats, axis=0)
    return out[:max_samples]


def compute_fid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sigma_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(np.real(fid))


def compute_kid(real_feats: np.ndarray, gen_feats: np.ndarray) -> float:
    # Unbiased polynomial-kernel MMD estimate with fixed degree-3 kernel.
    x = real_feats
    y = gen_feats
    d = x.shape[1]

    k_xx = ((x @ x.T) / d + 1.0) ** 3
    k_yy = ((y @ y.T) / d + 1.0) ** 3
    k_xy = ((x @ y.T) / d + 1.0) ** 3

    n = x.shape[0]
    m = y.shape[0]
    kid = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2.0 * k_xy.mean()
    )
    return float(kid)


def nearest_neighbor_l2(generated: torch.Tensor, train_batch: torch.Tensor) -> float:
    g = generated.flatten(1)
    t = train_batch.flatten(1)
    dists = torch.cdist(g, t, p=2)
    nearest = dists.min(dim=1).values
    return float(nearest.mean().item())


def estimate_bpd(
    model: nn.Module,
    data_loader: Iterable,
    state: ScheduleState,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    model.eval()
    total_elbo = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_idx, (x0, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            x0 = x0.to(device)
            b = x0.size(0)

            t = sample_timesteps(b, state, device=device)
            xt, eps = q_sample(x0, t, state)
            eps_hat = model(xt, t)

            x0_dim = float(x0[0].numel())

            mu_q, var_q = q_posterior_mean_var(x0, xt, t, state)
            mu_p = p_mean_from_eps(xt, t, eps_hat, state)
            var_p = var_q

            kl_t = 0.5 * ((mu_q - mu_p) ** 2 / torch.clamp(var_p, min=1e-12)).flatten(1).sum(dim=1)
            diffusion_term = state.num_timesteps * kl_t

            alpha_bar_final = state.alpha_bars[state.num_timesteps if state.one_based_indexing else state.num_timesteps - 1]
            var_l = torch.clamp(1.0 - alpha_bar_final, min=1e-12)
            prior_term = 0.5 * (
                var_l + alpha_bar_final * (x0.flatten(1) ** 2) - 1.0 - torch.log(var_l)
            ).sum(dim=1)

            elbo = prior_term + diffusion_term
            bpd = elbo / (x0_dim * np.log(2.0))

            total_elbo += float(bpd.sum().item())
            total_count += b

    return total_elbo / max(total_count, 1)
