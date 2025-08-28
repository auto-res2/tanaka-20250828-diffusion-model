import os
from typing import Tuple

import numpy as np
import torch


def index_to_logsnr(alphas_cumprod: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    a2 = alphas_cumprod[t_idx]
    return torch.log(a2 / (1 - a2 + 1e-8))


def compute_expert_timestep_heatmap(model, alphas_cumprod: torch.Tensor, device: str, batches: int = 5, batch_size: int = 16):
    # Returns counts per (expert, timestep_bin)
    if not hasattr(model, 'blocks') or len(getattr(model, 'blocks', [])) == 0:
        raise ValueError('Model must contain blocks for visualization.')
    first_block = model.blocks[0]
    M = int(getattr(first_block, 'M', 0))
    if M <= 0:
        experts = getattr(first_block, 'experts', None)
        if experts is not None:
            M = len(experts)
    if M <= 0:
        raise ValueError('Could not determine number of experts (M).')

    bins = 10
    heat = np.zeros((M, bins), dtype=np.float64)
    img_size = model.img_size
    for _ in range(batches):
        x0 = torch.rand(batch_size, 3, img_size, img_size, device=device)
        t_idx = torch.randint(0, alphas_cumprod.numel(), (batch_size,), device=device)
        logsnr = index_to_logsnr(alphas_cumprod, t_idx)
        with torch.no_grad():
            _ = model(x0, logsnr)
        # take layer 0 selections
        sel = getattr(model.blocks[0], '_last_selection', None)  # [B,N,M]
        if sel is None:
            continue
        e_ids = sel.argmax(dim=-1)  # [B,N]
        tbin = torch.clamp((t_idx.float() / alphas_cumprod.numel() * bins).long(), 0, bins - 1)
        for b in range(min(batch_size, e_ids.size(0))):
            eb = e_ids[b].view(-1).detach().cpu().numpy()
            tb = int(tbin[b].item())
            for e in eb:
                heat[e, tb] += 1
    return heat


def evaluate_router_stability(model, alphas_cumprod: torch.Tensor, device: str, batch_size: int = 8) -> float:
    if not hasattr(model, 'blocks') or len(getattr(model, 'blocks', [])) == 0:
        return 0.0
    model.eval()
    with torch.no_grad():
        x = torch.rand(batch_size, 3, model.img_size, model.img_size, device=device)
        t_idx = torch.randint(0, alphas_cumprod.numel(), (batch_size,), device=device)
        logsnr = index_to_logsnr(alphas_cumprod, t_idx)
        _ = model(x, logsnr)
        base = [getattr(blk, '_last_selection', None) for blk in model.blocks]
        base = [b.argmax(dim=-1) for b in base if b is not None]
        if not base:
            return 0.0
        jitter = torch.clamp(x + 0.01 * torch.randn_like(x), 0.0, 1.0)
        _ = model(jitter, logsnr)
        jit = [getattr(blk, '_last_selection', None) for blk in model.blocks]
        jit = [j.argmax(dim=-1) for j in jit if j is not None]
        diffs = [(b != j).float().mean().item() for b, j in zip(base, jit)]
        return float(np.mean(diffs)) if diffs else 0.0
