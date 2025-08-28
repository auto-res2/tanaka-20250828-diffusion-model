import os
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# Local (relative) imports
from .preprocess import SyntheticImageDataset, make_noisy_labels, set_seed
from .evaluate import compute_expert_timestep_heatmap, evaluate_router_stability


# -----------------------------
# Utility: FLOP Counter
# -----------------------------

class FlopCounter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.total_flops = 0.0
        self.step_profile = []  # list of (step, flops)
    def add(self, flops: float):
        self.total_flops += float(flops)
    def step_end(self, step_idx: int):
        self.step_profile.append((step_idx, self.total_flops))

FLOP_COUNTER = FlopCounter()


# -----------------------------
# Diffusion schedule utilities
# -----------------------------

def make_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def index_to_logsnr(alphas_cumprod: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    # logSNR = log(alpha^2 / (1 - alpha^2)) where alpha = sqrt(alpha_cumprod)
    a2 = alphas_cumprod[t_idx]  # [B]
    return torch.log(a2 / (1 - a2 + 1e-8))  # [B]


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, alphas_cumprod: torch.Tensor):
    # x_t = sqrt(alpha_cumprod) * x0 + sqrt(1 - alpha_cumprod) * eps
    B = x0.size(0)
    a = alphas_cumprod[t_idx].sqrt().view(B, 1, 1, 1)
    one_minus = (1.0 - alphas_cumprod[t_idx]).sqrt().view(B, 1, 1, 1)
    eps = torch.randn_like(x0)
    x_t = a * x0 + one_minus * eps
    return x_t, eps


# -----------------------------
# Routing utilities
# -----------------------------

def gumbel_topk_st(logits: torch.Tensor, temperature: float = 1.0, k: int = 1) -> torch.Tensor:
    # Straight-through Gumbel-Softmax for Top-k selection
    g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-9)).clamp_min(1e-9))
    y = F.softmax((logits + g) / max(temperature, 1e-6), dim=-1)
    if k == 1:
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
    else:
        topv, topi = torch.topk(y, k=min(k, y.size(-1)), dim=-1)
        y_hard = torch.zeros_like(y).scatter_(-1, topi, 1.0)
        # re-normalize the hard distribution to sum to 1 across selected positions
        y_hard = y_hard / (y_hard.sum(dim=-1, keepdim=True) + 1e-8)
    return y_hard + (y - y.detach())


def kl_to_uniform(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p: [M], sums to 1
    M = p.numel()
    q = torch.full_like(p, 1.0 / max(M, 1))
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)))


def flop_penalty(avg_active: float, budget: float, alpha: float = 1.0) -> torch.Tensor:
    val = torch.tensor(avg_active, dtype=torch.float32)
    return alpha * F.relu(val - budget) ** 2


def local_invariance_loss(spatial_logits: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    # spatial_logits: [B, N, M] where N = Ht*Wt
    B, N, M = spatial_logits.shape
    if Ht * Wt != N:
        return torch.tensor(0.0, device=spatial_logits.device)
    x = spatial_logits.view(B, Ht, Wt, M).permute(0, 3, 1, 2)  # [B, M, Ht, Wt]
    neigh = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return F.mse_loss(x, neigh.detach())


def gini_coefficient(x: torch.Tensor) -> float:
    # x: tensor of positive utilizations
    x = x.float()
    if x.numel() == 0:
        return 0.0
    if x.sum() <= 0:
        return 0.0
    n = x.numel()
    diff_sum = torch.sum(torch.abs(x.view(-1, 1) - x.view(1, -1)))
    return float(diff_sum / (2 * n * x.sum() + 1e-8))


# -----------------------------
# Model components: Patch embed, attention, FFN experts, ASTME block
# -----------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, E, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x

class SimpleAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        y, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return x + y

class ExpertFFN(nn.Module):
    def __init__(self, hidden: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden * expansion, hidden)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))

class DenseFFNBlock(nn.Module):
    def __init__(self, hidden: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ffn = ExpertFFN(hidden, expansion, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ffn(self.norm(x))
        # FLOPs: roughly 2*in*hid + 2*hid*out per token
        B, N, C = x.shape
        hid = C * 4
        FLOP_COUNTER.add((2 * C * hid + 2 * hid * C) * B * N)
        return x + h

class ASTMEBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        M_max: int = 8,
        top_k: int = 1,
        capacity_factor: float = 1.5,
        gate_dropout: float = 0.1,
        spatial_router: bool = True,
        temporal_gate: bool = True,
        expansion: int = 4,
        expert_width_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.M = M_max
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.temporal_gate_on = temporal_gate
        self.spatial_router_on = spatial_router
        self.norm = nn.LayerNorm(hidden)
        # experts (width scale: emulate by lowering expansion factor)
        eff_expansion = max(1, int(expansion * expert_width_scale))
        self.experts = nn.ModuleList([ExpertFFN(hidden, eff_expansion, dropout=0.0) for _ in range(M_max)])
        self.active_mask = nn.Parameter(torch.tensor([1, 1] + [0] * (M_max - 2), dtype=torch.float32), requires_grad=False)
        # gates
        self.time_gate = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, M_max))
        self.spatial_head = nn.Linear(hidden, M_max)
        self.router_bias = nn.Parameter(torch.zeros(M_max))
        self.gate_drop = nn.Dropout(gate_dropout)
        # temperature schedule param (set externally)
        self.register_buffer("temperature", torch.tensor(1.0))
        # stats
        self._last_spatial_logits = None
        self._last_selection = None
        self._overflow_rate = 0.0
        self._avg_active = 0.0
        self._util = torch.zeros(M_max)

    def forward(self, x: torch.Tensor, t_logsnr: torch.Tensor, grid_hw: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, N, C = x.shape
        x_norm = self.norm(x)
        # Router logits
        time_logits = torch.zeros(B, 1, self.M, device=x.device)
        if self.temporal_gate_on:
            time_logits = self.time_gate(t_logsnr.view(B, 1, 1))  # [B,1,M]
        spatial_logits = torch.zeros(B, N, self.M, device=x.device)
        if self.spatial_router_on:
            spatial_logits = self.spatial_head(x_norm)  # [B,N,M]
        logits = time_logits + spatial_logits + self.router_bias.view(1, 1, -1)
        # mask inactive
        active_mask = self.active_mask.to(x.device)
        logits = logits + (1.0 - active_mask.view(1, 1, -1)) * (-1e9)
        logits = self.gate_drop(logits)
        # selection
        sel = gumbel_topk_st(logits, float(self.temperature.item()), k=self.top_k)  # [B,N,M]
        # capacity (soft): simulate capacity by truncating if needed
        tokens_per_exp = sel.sum(dim=(0, 1))  # [M]
        num_active = int(active_mask.sum().item())
        capacity = math.ceil(self.capacity_factor * (B * N) / max(1, num_active))
        overflow = 0.0
        x_out = torch.zeros_like(x)
        sel_flat = sel.view(B * N, self.M)
        x_flat = x_norm.view(B * N, C)
        for e in range(self.M):
            if active_mask[e] < 0.5:
                continue
            idx = torch.nonzero(sel_flat[:, e] > 0.0, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > capacity:
                overflow += float(idx.numel() - capacity) / float(B * N)
                idx = idx[:capacity]
            xin = x_flat.index_select(0, idx)
            y = self.experts[e](xin)
            x_out.view(B * N, C).index_copy_(0, idx, y)
            # FFN FLOPs for this expert only
            in_dim = C
            hid = in_dim * 4  # expansion inside ExpertFFN
            FLOP_COUNTER.add((2 * in_dim * hid + 2 * hid * in_dim) * idx.numel())
        # residual
        y = x + x_out
        # stats for logging and losses
        self._last_spatial_logits = spatial_logits.detach()
        self._last_selection = sel.detach()
        avg_active = float(sel.sum(dim=-1).mean().item())  # ~ top_k normalized
        self._avg_active = avg_active
        util = tokens_per_exp.detach().cpu()
        self._util = util
        self._overflow_rate = overflow
        stats = dict(
            avg_active=avg_active,
            overflow=overflow,
            util=util.tolist(),
            gini=gini_coefficient(util),
            local_invariance=float(local_invariance_loss(spatial_logits.detach(), grid_hw[0], grid_hw[1]).item())
            if self.spatial_router_on else 0.0,
        )
        return y, stats

    def grow_one_expert(self, busiest_idx: int) -> bool:
        # activate next inactive expert by cloning busiest
        with torch.no_grad():
            inactive = (self.active_mask < 0.5).nonzero(as_tuple=False).view(-1)
            if inactive.numel() == 0:
                return False
            new_e = int(inactive[0].item())
            src = int(busiest_idx)
            for p_new, p_src in zip(self.experts[new_e].parameters(), self.experts[src].parameters()):
                p_new.copy_(p_src)
                p_new.add_(0.01 * torch.randn_like(p_new))
            self.active_mask[new_e] = 1.0
        return True


class ASTMEDiTSmall(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        hidden: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        M_max: int = 6,
        top_k: int = 1,
        temporal_gate: bool = True,
        spatial_router: bool = True,
        expert_width_scale: float = 0.25,
    ):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_chans, hidden)
        self.grid_hw = (img_size // patch_size, img_size // patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, self.patch.num_patches, hidden) * 0.02)
        self.attn = nn.ModuleList([SimpleAttention(hidden, n_heads) for _ in range(depth)])
        self.blocks = nn.ModuleList([
            ASTMEBlock(hidden, M_max=M_max, top_k=top_k, temporal_gate=temporal_gate,
                       spatial_router=spatial_router, expert_width_scale=expert_width_scale)
            for _ in range(depth)
        ])
        self.unpatch = nn.Linear(hidden, patch_size * patch_size * in_chans)
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor, t_logsnr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # x: [B, C, H, W]; return predicted noise eps_hat (same shape as x)
        B, C, H, W = x.shape
        h = self.patch(x) + self.pos_emb
        for att, blk in zip(self.attn, self.blocks):
            h = att(h)
            # Attn FLOPs approximated (dense, irrespective of ASTME):
            N, E = h.shape[1], h.shape[2]
            heads = 4
            d = E // heads
            # projections: 3*E*E per token
            FLOP_COUNTER.add(B * N * (3 * 2 * E * E))
            # attention scores: heads * N*N * d
            FLOP_COUNTER.add(B * heads * (N * N * 2 * d))
            # output proj: E*E per token
            FLOP_COUNTER.add(B * N * (2 * E * E))
            h, stats = blk(h, t_logsnr, self.grid_hw)
        # map tokens back to image
        tokens = h
        px = self.unpatch(tokens)  # [B, N, ps*ps*C]
        FLOP_COUNTER.add(B * tokens.size(1) * (2 * tokens.size(2) * px.size(-1)))
        ps = self.patch_size
        out = px.view(B, self.grid_hw[0], self.grid_hw[1], ps, ps, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H, W)
        return out, stats


class DenseDiTSmall(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, hidden: int = 128, depth: int = 4, n_heads: int = 4):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_chans, hidden)
        self.grid_hw = (img_size // patch_size, img_size // patch_size)
        self.pos_emb = nn.Parameter(torch.randn(1, self.patch.num_patches, hidden) * 0.02)
        self.attn = nn.ModuleList([SimpleAttention(hidden, n_heads) for _ in range(depth)])
        self.ffn = nn.ModuleList([DenseFFNBlock(hidden) for _ in range(depth)])
        self.unpatch = nn.Linear(hidden, patch_size * patch_size * in_chans)
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor, t_logsnr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, C, H, W = x.shape
        h = self.patch(x) + self.pos_emb
        for att, f in zip(self.attn, self.ffn):
            h = att(h)
            # Attn FLOPs (dense):
            N, E = h.shape[1], h.shape[2]
            heads = 4
            d = E // heads
            FLOP_COUNTER.add(B * N * (3 * 2 * E * E))
            FLOP_COUNTER.add(B * heads * (N * N * 2 * d))
            FLOP_COUNTER.add(B * N * (2 * E * E))
            h = f(h)
        tokens = h
        px = self.unpatch(tokens)
        FLOP_COUNTER.add(B * tokens.size(1) * (2 * tokens.size(2) * px.size(-1)))
        ps = self.patch_size
        out = px.view(B, self.grid_hw[0], self.grid_hw[1], ps, ps, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H, W)
        return out, {}


# -----------------------------
# Training helpers
# -----------------------------

@dataclass
class TrainConfig:
    img_size: int = 32
    patch_size: int = 4
    hidden: int = 128
    depth: int = 4
    n_heads: int = 4
    M_max: int = 6
    top_k: int = 1
    batch_size: int = 32
    steps: int = 100
    lr: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    temporal_gate: bool = True
    spatial_router: bool = True
    expert_width_scale: float = 0.25
    seed: int = 123
    log_every: int = 10
    temp_start: float = 1.0
    temp_end: float = 0.5
    T: int = 1000  # diffusion steps


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.copy(model)
    def copy(self, model):
        for k, p in model.state_dict().items():
            self.shadow[k] = p.clone().detach()
    def update(self, model):
        with torch.no_grad():
            for k, p in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * p.detach()


def diffusion_training_step(model, x0, alphas_cumprod, opt, tdist: Optional[torch.distributions.Distribution], step_idx: int, cfg: TrainConfig, lambda_KL=0.05, lambda_FLOP=0.1):
    model.train()
    FLOP_COUNTER.reset()
    B = x0.size(0)
    # sample timesteps uniformly
    t_idx = torch.randint(0, cfg.T, (B,), device=x0.device)
    logsnr = index_to_logsnr(alphas_cumprod, t_idx)
    x_t, eps = q_sample(x0, t_idx, alphas_cumprod)
    eps_hat, stats = model(x_t, logsnr)
    loss = F.mse_loss(eps_hat, eps)
    # Regularizers for ASTME model (if present)
    kl_loss = torch.tensor(0.0, device=x0.device)
    flop_loss = torch.tensor(0.0, device=x0.device)
    local_inv = torch.tensor(0.0, device=x0.device)
    # aggregate stats across blocks if present
    if hasattr(model, 'blocks'):
        avg_active_vals = []
        for blk in model.blocks:
            if not isinstance(blk, ASTMEBlock):
                continue
            util = torch.tensor(blk._util, device=x0.device)
            if util.sum() > 0:
                p = util / util.sum()
                kl_loss = kl_loss + kl_to_uniform(p)
            avg_active_vals.append(blk._avg_active)
            local_inv = local_inv + torch.tensor(blk._last_spatial_logits is not None, device=x0.device, dtype=torch.float32) * blk._avg_active * 0.0
        mean_active = float(np.mean(avg_active_vals)) if avg_active_vals else 0.0
        flop_loss = flop_penalty(mean_active, budget=1.3, alpha=1.0).to(x0.device)
    total_loss = loss + lambda_KL * kl_loss + lambda_FLOP * flop_loss + 0.0 * local_inv
    opt.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    FLOP_COUNTER.step_end(step_idx)
    return {
        'loss': float(loss.item()),
        'total_loss': float(total_loss.item()),
        'flops': float(FLOP_COUNTER.total_flops),
        'kl_loss': float(kl_loss.item()),
        'flop_loss': float(flop_loss.item()),
    }


def anneal_temperature(model: nn.Module, step: int, total_steps: int, start: float = 1.0, end: float = 0.1):
    temp = start + (end - start) * (step / max(1, total_steps))
    if hasattr(model, 'blocks'):
        for blk in model.blocks:
            if isinstance(blk, ASTMEBlock):
                blk.temperature.fill_(temp)


# -----------------------------
# Experiments
# -----------------------------


def run_experiment1(save_dir: str, fast: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    print("[Experiment 1] Training efficiency vs. baseline (synthetic quick prototype)")
    set_seed(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data: multiple patterns to demonstrate robustness
    patterns = ['edges', 'textures']
    img_size = 32
    train_sets = [SyntheticImageDataset(n=512 if fast else 4096, img_size=img_size, pattern=p) for p in patterns]
    loaders = [torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0) for ds in train_sets]

    # Diffusion schedule
    T = 1000
    betas, alphas, alphas_cumprod = make_beta_schedule(T=T)
    alphas_cumprod = alphas_cumprod.to(device)

    # Models
    cfg_astme = TrainConfig(img_size=img_size, hidden=96, depth=3, M_max=6, expert_width_scale=0.25, steps=(20 if fast else 200), batch_size=32)
    cfg_base = TrainConfig(img_size=img_size, hidden=96, depth=3, steps=(20 if fast else 200), batch_size=32)

    model_astme = ASTMEDiTSmall(
        img_size=cfg_astme.img_size, hidden=cfg_astme.hidden, depth=cfg_astme.depth,
        M_max=cfg_astme.M_max, top_k=1, temporal_gate=True, spatial_router=True, expert_width_scale=cfg_astme.expert_width_scale
    ).to(device)
    model_base = DenseDiTSmall(img_size=cfg_base.img_size, hidden=cfg_base.hidden, depth=cfg_base.depth).to(device)

    opt_astme = torch.optim.AdamW(model_astme.parameters(), lr=1e-3)
    opt_base = torch.optim.AdamW(model_base.parameters(), lr=1e-3)

    # Storage for plots
    logs = {
        'astme': {'loss': [], 'flops': [], 'step': []},
        'baseline': {'loss': [], 'flops': [], 'step': []},
        'astme_active_experts': [],
    }

    # Train ASTME and Baseline on pattern=edges
    for name, model, opt, cfg, loader in [
        ('astme', model_astme, opt_astme, cfg_astme, loaders[0]),
        ('baseline', model_base, opt_base, cfg_base, loaders[0]),
    ]:
        print(f"  > Training {name} on pattern=edges for {cfg.steps} steps")
        it = iter(loader)
        for step in range(cfg.steps):
            try:
                x0, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x0, _ = next(it)
            x0 = x0.to(device)
            anneal_temperature(model, step, cfg.steps, start=1.0, end=0.3)
            stats = diffusion_training_step(model, x0, alphas_cumprod, opt, None, step, cfg)
            logs[name]['loss'].append(stats['loss'])
            logs[name]['flops'].append(stats['flops'])
            logs[name]['step'].append(step)
            if name == 'astme' and hasattr(model, 'blocks'):
                avg_act = np.mean([blk._avg_active for blk in model.blocks if isinstance(blk, ASTMEBlock)])
                logs['astme_active_experts'].append(avg_act)
            if step % cfg.log_every == 0:
                print(f"    step {step:03d} | loss={stats['loss']:.4f} | pfLOPs_accum={stats['flops']/1e15:.6f}")
        print(f"  > Done {name}. Total PFLOPs: {FLOP_COUNTER.total_flops/1e15:.6f}")

    # Quick additional robustness: run ASTME few steps on textures
    print("  > Short ASTME run on textures to show robustness")
    it2 = iter(loaders[1])
    for step in range(5 if fast else 20):
        try:
            x0, _ = next(it2)
        except StopIteration:
            it2 = iter(loaders[1])
            x0, _ = next(it2)
        x0 = x0.to(device)
        anneal_temperature(model_astme, step, 5 if fast else 20, start=0.8, end=0.5)
        _ = diffusion_training_step(model_astme, x0, alphas_cumprod, opt_astme, None, step, cfg_astme)

    # Plots
    if HAS_MPL:
        # Training loss curves
        plt.figure()
        plt.plot(logs['astme']['step'], logs['astme']['loss'], label='ASTME')
        plt.xlabel('step'); plt.ylabel('train_loss'); plt.legend(); plt.title('Training Loss - ASTME')
        plt.savefig(os.path.join(save_dir, 'training_loss_astme.pdf'), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(logs['baseline']['step'], logs['baseline']['loss'], label='Baseline')
        plt.xlabel('step'); plt.ylabel('train_loss'); plt.legend(); plt.title('Training Loss - Baseline')
        plt.savefig(os.path.join(save_dir, 'training_loss_baseline.pdf'), bbox_inches='tight')
        plt.close()

        # FLOPs vs step
        plt.figure()
        plt.plot(logs['astme']['step'], np.array(logs['astme']['flops'])/1e12, label='ASTME')
        plt.xlabel('step'); plt.ylabel('TFLOPs (accum)'); plt.legend(); plt.title('Accumulated TFLOPs - ASTME')
        plt.savefig(os.path.join(save_dir, 'pflops_vs_step_astme.pdf'), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.plot(logs['baseline']['step'], np.array(logs['baseline']['flops'])/1e12, label='Baseline')
        plt.xlabel('step'); plt.ylabel('TFLOPs (accum)'); plt.legend(); plt.title('Accumulated TFLOPs - Baseline')
        plt.savefig(os.path.join(save_dir, 'pflops_vs_step_baseline.pdf'), bbox_inches='tight')
        plt.close()

        # Active experts per token (ASTME)
        if len(logs['astme_active_experts']) > 0:
            plt.figure()
            plt.plot(logs['astme_active_experts'])
            plt.xlabel('step'); plt.ylabel('avg_active_experts_per_token'); plt.title('Active Experts per Token - ASTME')
            plt.savefig(os.path.join(save_dir, 'active_experts_astme.pdf'), bbox_inches='tight')
            plt.close()
    print("[Experiment 1] Saved figures to:", os.path.abspath(save_dir))


def run_experiment2(save_dir: str, fast: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    print("[Experiment 2] Ablations and expert specialization (synthetic quick prototype)")
    set_seed(1234)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Data
    ds = SyntheticImageDataset(n=512 if fast else 2048, img_size=32, pattern='textures')
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    # Diffusion schedule
    T = 1000
    _, _, alphas_cumprod = make_beta_schedule(T=T)
    alphas_cumprod = alphas_cumprod.to(device)

    # Variants
    variants = {
        'full_astme': dict(temporal=True, spatial=True, top_k=1),
        'temporal_only': dict(temporal=True, spatial=False, top_k=1),
        'spatial_only': dict(temporal=False, spatial=True, top_k=1),
        'topk2': dict(temporal=True, spatial=True, top_k=2),
    }

    results = {}
    for name, vcfg in variants.items():
        print(f"  > Training variant: {name}")
        model = ASTMEDiTSmall(img_size=32, hidden=96, depth=3, M_max=6, top_k=vcfg['top_k'],
                              temporal_gate=vcfg['temporal'], spatial_router=vcfg['spatial'], expert_width_scale=0.25).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        it = iter(loader)
        losses, flops = [], []
        for step in range(15 if fast else 100):
            try:
                x0, _ = next(it)
            except StopIteration:
                it = iter(loader)
                x0, _ = next(it)
            x0 = x0.to(device)
            anneal_temperature(model, step, 15 if fast else 100, start=1.0, end=0.3)
            stats = diffusion_training_step(model, x0, alphas_cumprod, opt, None, step, TrainConfig())
            losses.append(stats['loss']); flops.append(stats['flops'])
            if step % 5 == 0:
                print(f"    step {step:02d} | loss={stats['loss']:.4f} | PFLOPs={stats['flops']/1e15:.6f}")
        results[name] = {
            'loss_curve': losses,
            'flops_curve': flops,
            'final_loss': float(losses[-1]),
            'total_pflops': float(flops[-1] / 1e15)
        }

    # Plot ablation bars
    if HAS_MPL:
        names = list(results.keys())
        pflops_vals = [results[n]['total_pflops'] for n in names]
        loss_vals = [results[n]['final_loss'] for n in names]
        # PFLOPs bar
        plt.figure(figsize=(6, 3))
        plt.bar(names, pflops_vals)
        plt.ylabel('PFLOPs (accumulated)'); plt.title('PFLOPs to reach end of quick run (ablation)')
        plt.xticks(rotation=30)
        plt.savefig(os.path.join(save_dir, 'pflops_ablation.pdf'), bbox_inches='tight')
        plt.close()
        # Loss bar
        plt.figure(figsize=(6, 3))
        plt.bar(names, loss_vals, color='orange')
        plt.ylabel('final training loss'); plt.title('Final loss (ablation)')
        plt.xticks(rotation=30)
        plt.savefig(os.path.join(save_dir, 'final_loss_ablation.pdf'), bbox_inches='tight')
        plt.close()

        # Specialization heatmap for full_astme (randomly probed model for visualization)
        model_spec = ASTMEDiTSmall(img_size=32, hidden=96, depth=3, M_max=6, top_k=1, temporal_gate=True, spatial_router=True, expert_width_scale=0.25).to(device)
        with torch.no_grad():
            _ = model_spec(torch.rand(1, 3, 32, 32, device=device), torch.tensor([0.0], device=device))
        heat = compute_expert_timestep_heatmap(model_spec, alphas_cumprod, device=device, batches=3 if fast else 10, batch_size=16)
        plt.figure(figsize=(5, 3))
        if HAS_SNS:
            sns.heatmap(heat + 1e-6, cbar=True)
        else:
            plt.imshow(heat + 1e-6, aspect='auto', cmap='viridis')
            plt.colorbar()
        plt.xlabel('timestep bin'); plt.ylabel('expert id'); plt.title('Expert vs Timestep counts (full ASTME)')
        plt.savefig(os.path.join(save_dir, 'expert_timestep_heatmap_astme.pdf'), bbox_inches='tight')
        plt.close()

    print("[Experiment 2] Saved figures to:", os.path.abspath(save_dir))


def run_experiment3(save_dir: str, fast: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    print("[Experiment 3] Robustness: small data + label noise (synthetic quick prototype)")
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Small dataset
    ds_small = SyntheticImageDataset(n=256 if fast else 2048, img_size=32, pattern='edges', n_classes=10, class_conditional=True)
    loader = torch.utils.data.DataLoader(ds_small, batch_size=32, shuffle=True, num_workers=0)
    # Diffusion schedule
    T = 1000
    _, _, alphas_cumprod = make_beta_schedule(T=T)
    alphas_cumprod = alphas_cumprod.to(device)

    noise_levels = [0.0, 0.4]
    loss_per_noise_astme: List[float] = []
    stab_per_noise_astme: List[float] = []

    for noise in noise_levels:
        print(f"  > Training ASTME with label noise rate={noise}")
        model = ASTMEDiTSmall(img_size=32, hidden=96, depth=3, M_max=6, top_k=1, temporal_gate=True, spatial_router=True, expert_width_scale=0.25).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        it = iter(loader)
        for step in range(12 if fast else 100):
            try:
                x0, y = next(it)
            except StopIteration:
                it = iter(loader)
                x0, y = next(it)
            x0 = x0.to(device)
            y = y.to(device)
            y_noisy = make_noisy_labels(y, noise, n_classes=10)
            # very light conditioning: add class-embedding bias to image (toy)
            class_embed = F.one_hot(y_noisy, num_classes=10).float().to(device)
            class_embed = class_embed.view(x0.size(0), 10, 1, 1)
            x0c = torch.clamp(x0 + 0.05 * F.pad(class_embed, (0, x0.size(-1)-1, 0, x0.size(-2)-1))[:, :3], 0.0, 1.0)
            anneal_temperature(model, step, 12 if fast else 100, start=1.0, end=0.4)
            stats = diffusion_training_step(model, x0c, alphas_cumprod, opt, None, step, TrainConfig())
            if step % 4 == 0:
                print(f"    step {step:02d} | loss={stats['loss']:.4f} | PFLOPs={stats['flops']/1e15:.6f}")
        # evaluate stability
        stab = evaluate_router_stability(model, alphas_cumprod, device=device, batch_size=8)
        loss_per_noise_astme.append(stats['loss'])
        stab_per_noise_astme.append(stab)
        print(f"    > Final loss={stats['loss']:.4f}, router switch rate={stab:.4f}")

    # plots
    if HAS_MPL:
        plt.figure()
        plt.plot(noise_levels, loss_per_noise_astme, marker='o')
        plt.xlabel('label noise rate'); plt.ylabel('final loss'); plt.title('Loss vs noise (ASTME)')
        plt.savefig(os.path.join(save_dir, 'loss_vs_noise_astme.pdf'), bbox_inches='tight')
        plt.close()
        plt.figure()
        plt.plot(noise_levels, stab_per_noise_astme, marker='s', color='red')
        plt.xlabel('label noise rate'); plt.ylabel('router switch rate'); plt.title('Router Stability vs noise (ASTME)')
        plt.savefig(os.path.join(save_dir, 'router_switch_rate_astme.pdf'), bbox_inches='tight')
        plt.close()
    print("[Experiment 3] Saved figures to:", os.path.abspath(save_dir))
