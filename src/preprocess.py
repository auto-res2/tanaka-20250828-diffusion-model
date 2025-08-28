import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ------------------------------
# Synthetic dataset with multiple patterns (robustness)
# ------------------------------

class SyntheticPatterns(Dataset):
    """
    Returns tuples (x0, cond_vec, cls_label, prompt_id)
      - x0: ground-truth image in [-1, 1]
      - cond_vec: conditioning vector (pattern-type + parameters)
      - cls_label: int in {0..n_types-1}
      - prompt_id: integer id to allow canonical reconstruction

    Patterns: stripes, checkerboard, circle, gaussian_blobs
    """
    def __init__(self, n_samples=512, img_size=32, n_types=4, seed=0, canonical=False):
        super().__init__()
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_types = n_types
        self.rng = np.random.RandomState(seed)
        self.canonical = canonical
        self.samples = []
        for i in range(n_samples):
            cls = i % n_types
            prompt_id = i
            x0, cond = self._make_sample(cls=cls, prompt_id=prompt_id, canonical=canonical)
            self.samples.append((x0, cond, cls, prompt_id))

    def _make_sample(self, cls: int, prompt_id: int, canonical=False):
        H = W = self.img_size
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        rng = np.random.RandomState(prompt_id if canonical else prompt_id + 12345)
        img = np.zeros((H, W), dtype=np.float32)
        if cls == 0:
            angle = 0.0 if canonical else rng.uniform(0, math.pi)
            freq = 4 if canonical else rng.randint(3, 7)
            xx = grid_x * math.cos(angle) + grid_y * math.sin(angle)
            img = 0.5 * (np.sin(2 * math.pi * xx / (W / freq)) + 1.0)
            params = [angle / math.pi, freq / 10.0]
        elif cls == 1:
            k = 4 if canonical else rng.randint(3, 7)
            img = (((grid_x // (W // k)) + (grid_y // (H // k))) % 2).astype(np.float32)
            params = [k / 10.0, 0.0]
        elif cls == 2:
            rad = H // 4 if canonical else rng.randint(H // 6, H // 3)
            cx = W // 2 if canonical else rng.randint(W // 4, 3 * W // 4)
            cy = H // 2 if canonical else rng.randint(W // 4, 3 * W // 4)
            dist = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
            img = (dist <= rad ** 2).astype(np.float32)
            params = [rad / (H / 2), cx / W]
        else:
            n_blobs = 3 if canonical else rng.randint(2, 5)
            img = np.zeros((H, W), dtype=np.float32)
            for _ in range(n_blobs):
                cx = rng.uniform(0, W)
                cy = rng.uniform(0, H)
                sx = rng.uniform(W / 16, W / 8)
                sy = rng.uniform(H / 16, H / 8)
                g = np.exp(-(((grid_x - cx) ** 2) / (2 * sx ** 2) + ((grid_y - cy) ** 2) / (2 * sy ** 2)))
                img += g
            img = img / (img.max() + 1e-6)
            params = [n_blobs / 8.0, 0.0]
        if not canonical:
            img = np.clip(img + rng.normal(scale=0.05, size=img.shape), 0.0, 1.0)
        x0 = (img * 2.0 - 1.0).astype(np.float32)
        cond = np.zeros(8, dtype=np.float32)
        cond[cls] = 1.0
        cond[4:6] = np.array(params, dtype=np.float32)
        cond[6] = H / 64.0
        cond[7] = 1.0
        x0 = x0[None, ...]
        return x0, cond

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x0, cond, cls, pid = self.samples[idx]
        return torch.from_numpy(x0), torch.from_numpy(cond), torch.tensor(cls, dtype=torch.long), torch.tensor(pid, dtype=torch.long)


# ------------------------------
# Diffusion utilities (toy)
# ------------------------------

@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_bar: torch.Tensor
    sqrt_alphas_bar: torch.Tensor
    sqrt_one_minus_alphas_bar: torch.Tensor


def make_linear_schedule(T: int = 50, beta_start=1e-4, beta_end=0.02, device=None) -> DiffusionSchedule:
    device = device or get_device()
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_bar=alphas_bar,
        sqrt_alphas_bar=torch.sqrt(alphas_bar),
        sqrt_one_minus_alphas_bar=torch.sqrt(1.0 - alphas_bar),
    )


def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: DiffusionSchedule, noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = sched.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
    sqrt_omab = sched.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_omab * noise, noise


def ddim_step(x: torch.Tensor, eps: torch.Tensor, t: int, t_prev: int, sched: DiffusionSchedule):
    ab_t = sched.alphas_bar[t]
    ab_prev = sched.alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
    sqrt_ab_t = torch.sqrt(ab_t)
    sqrt_ab_prev = torch.sqrt(ab_prev)
    sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
    x0_pred = (x - sqrt_one_minus_ab_t * eps) / (sqrt_ab_t + 1e-9)
    dir_term = torch.sqrt(1.0 - ab_prev) * eps
    x_prev = sqrt_ab_prev * x0_pred + dir_term
    return x_prev


def ddpm_step(x: torch.Tensor, eps: torch.Tensor, t: int, sched: DiffusionSchedule):
    beta_t = sched.betas[t]
    alpha_t = sched.alphas[t]
    alpha_bar_t = sched.alphas_bar[t]
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bar_t)
    x0_pred = (x - sqrt_one_minus_ab * eps) / torch.sqrt(alpha_bar_t + 1e-9)
    mean = torch.sqrt(alpha_t) * x + (1 - alpha_t) * x0_pred
    if t > 0:
        noise = torch.randn_like(x)
        var = beta_t
        x_prev = mean + torch.sqrt(var) * noise
    else:
        x_prev = mean
    return x_prev


def dpmpp_heun_step(x: torch.Tensor, eps_fn, t: int, t_prev: int, sched: DiffusionSchedule, cond_emb):
    eps_t = eps_fn(x, t, cond_emb)
    x_euler = ddim_step(x, eps_t, t, t_prev, sched)
    eps_t_prev = eps_fn(x_euler, t_prev, cond_emb) if t_prev >= 0 else eps_t
    x_heun = 0.5 * (x_euler + ddim_step(x, eps_t_prev, t, t_prev, sched))
    return x_heun


# ------------------------------
# Embeddings and conditioning
# ------------------------------

class SinusoidalTimeEmbedding(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device)
        )
        args = t.float().unsqueeze(1) / (t.max().float().clamp(min=1.0))
        args = args * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class GlobalCondMLP(torch.nn.Module):
    def __init__(self, cond_dim=8, t_dim=64, out_dim=128):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(cond_dim + t_dim, 256), torch.nn.SiLU(),
            torch.nn.Linear(256, out_dim)
        )

    def forward(self, cond_vec: torch.Tensor, t_emb: torch.Tensor):
        x = torch.cat([cond_vec, t_emb], dim=-1)
        return self.proj(x)


# ------------------------------
# Dataloaders and canonical set
# ------------------------------

def create_dataloaders(n_train=512, n_val=128, img_size=32, batch_size=64, seed=0):
    ds_train = SyntheticPatterns(n_samples=n_train, img_size=img_size, seed=seed, canonical=False)
    ds_val = SyntheticPatterns(n_samples=n_val, img_size=img_size, seed=seed+1, canonical=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    return ds_train, ds_val, dl_train, dl_val


def make_canonical_set(n=64, img_size=32, seed=123):
    return SyntheticPatterns(n_samples=n, img_size=img_size, seed=seed, canonical=True)
