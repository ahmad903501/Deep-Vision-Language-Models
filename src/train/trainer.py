from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.diffusion.ddpm import ancestral_sample
from src.diffusion.forward import q_sample, sample_timesteps
from src.diffusion.posterior import q_posterior_mean_var
from src.diffusion.schedule import ScheduleState
from src.eval.visualization import plot_training_loss, save_tensor_grid, save_trajectory_grids
from src.utils.config import AppConfig
from src.utils.io import ensure_dir, save_json


@dataclass
class TrainArtifacts:
    losses: List[float]
    final_checkpoint: Path
    overfit_final_loss: float
    posterior_check: Dict[str, float]


class DDPMTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        state: ScheduleState,
        train_loader,
        val_loader,
        config: AppConfig,
        device: torch.device,
        run_dir: Path,
        image_shape: tuple[int, int, int],
    ) -> None:
        self.model = model
        self.state = state
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.run_dir = run_dir
        self.image_shape = image_shape
        self.amp_enabled = config.training.amp and device.type == "cuda"
        self.amp_device_type = "cuda" if device.type == "cuda" else "cpu"

        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.amp_enabled)
        self.losses: List[float] = []

        ensure_dir(self.run_dir / "checkpoints")
        ensure_dir(self.run_dir / "images")
        ensure_dir(self.run_dir / "plots")
        ensure_dir(self.run_dir / "reports")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.config.training.optimizer.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _save_checkpoint(self, step: int) -> Path:
        ckpt_path = self.run_dir / "checkpoints" / f"step_{step:07d}.pt"
        payload = {
            "model": self._model_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "config": self.config.to_dict(),
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def _model_state_dict(self) -> Dict[str, torch.Tensor]:
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def run_overfit_gate(self, steps: int = 600) -> float:
        self.model.train()
        batch, _ = next(iter(self.train_loader))
        subset = batch[: self.config.training.overfit_subset_size].to(self.device)

        for _ in tqdm(range(steps), desc="Overfit gate", leave=False):
            t = sample_timesteps(subset.size(0), self.state, self.device)
            xt, eps = q_sample(subset, t, self.state)
            eps_hat = self.model(xt, t)
            loss = F.mse_loss(eps_hat, eps)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
            self.optimizer.step()

        return float(loss.item())

    def run_one_step_posterior_check(self, num_trials: int = 64) -> Dict[str, float]:
        self.model.eval()
        x0, _ = next(iter(self.val_loader))
        x0 = x0[:num_trials].to(self.device)

        min_t = 2 if self.state.one_based_indexing else 1
        max_t = self.state.num_timesteps
        t = torch.randint(min_t, max_t + 1, (x0.size(0),), device=self.device)

        xt, _ = q_sample(x0, t, self.state)
        mu_q, var_q = q_posterior_mean_var(x0, xt, t, self.state)
        x_prev = mu_q + torch.sqrt(torch.clamp(var_q, min=0.0)) * torch.randn_like(xt)

        dist_prev = ((x_prev - x0) ** 2).flatten(1).mean(dim=1)
        dist_curr = ((xt - x0) ** 2).flatten(1).mean(dim=1)

        lhs = float(dist_prev.mean().item())
        rhs = float(dist_curr.mean().item())

        return {
            "mean_dist_x_prev_to_x0": lhs,
            "mean_dist_xt_to_x0": rhs,
            "passed": float(lhs < rhs),
        }

    def train(self) -> TrainArtifacts:
        self.model.train()
        step = 0
        final_ckpt = self.run_dir / "checkpoints" / "final.pt"

        for epoch in range(self.config.training.num_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", leave=True)
            for x0, _ in pbar:
                if step >= self.config.training.max_steps:
                    break

                x0 = x0.to(self.device)
                t = sample_timesteps(x0.size(0), self.state, self.device)

                with torch.amp.autocast(self.amp_device_type, enabled=self.amp_enabled):
                    xt, eps = q_sample(x0, t, self.state)
                    eps_hat = self.model(xt, t)
                    loss = F.mse_loss(eps_hat, eps)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                step += 1
                self.losses.append(float(loss.item()))

                if step % self.config.training.log_every == 0:
                    pbar.set_postfix({"step": step, "loss": float(loss.item())})

                if step % self.config.training.sample_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        samples, _ = ancestral_sample(
                            model=self.model,
                            state=self.state,
                            shape=(self.config.sampling.num_samples, *self.image_shape),
                            device=self.device,
                            save_steps=self.config.sampling.save_trajectory_steps,
                        )
                    save_tensor_grid(
                        samples.cpu(),
                        self.run_dir / "images" / f"sample_grid_step_{step:07d}.png",
                    )
                    self.model.train()

                if step % self.config.training.save_every == 0:
                    self._save_checkpoint(step)

            if step >= self.config.training.max_steps:
                break

        torch.save(
            {
                "model": self._model_state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "config": self.config.to_dict(),
            },
            final_ckpt,
        )

        plot_training_loss(self.losses, self.run_dir / "plots" / "training_loss.png")

        self.model.eval()
        with torch.no_grad():
            final_samples, trajectory = ancestral_sample(
                model=self.model,
                state=self.state,
                shape=(self.config.sampling.num_samples, *self.image_shape),
                device=self.device,
                save_steps=self.config.sampling.save_trajectory_steps,
            )
        save_tensor_grid(final_samples.cpu(), self.run_dir / "images" / "final_samples_8x8.png")
        save_trajectory_grids(trajectory, self.run_dir / "images" / "trajectory")

        overfit_loss = self.run_overfit_gate()
        posterior_check = self.run_one_step_posterior_check()

        save_json(
            {
                "overfit_final_loss": overfit_loss,
                "posterior_check": posterior_check,
                "num_steps": step,
            },
            self.run_dir / "reports" / "sanity_gates.json",
        )

        return TrainArtifacts(
            losses=self.losses,
            final_checkpoint=final_ckpt,
            overfit_final_loss=overfit_loss,
            posterior_check=posterior_check,
        )
