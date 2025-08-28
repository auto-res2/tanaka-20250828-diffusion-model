import os
from typing import Tuple

import numpy as np
import torch

# Relative imports
from .train import index_to_logsnr, ASTMEDiTSmall, ASTMEBlock


def compute_expert_timestep_heatmap(model: ASTMEDiTSmall, alphas_cumprod: torch.Tensor, device: str, batches: int = 5, batch_size: int = 16):
    # Returns counts per (expert, timestep_bin)
    assert isinstance(model, ASTMEDiTSmall), "Model must be ASTMEDiTSmall for this visualization."
    assert hasattr(model, 'blocks') and isinstance(model.blocks[0], ASTMEBlock), "Model must contain ASTME blocks."
    M = model.blocks[0].M if isinstance(model.blocks[0], ASTMEBlock) else 0
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
        sel = model.blocks[0]._last_selection  # [B,N,M]
        if sel is None:
            continue
        e_ids = sel.argmax(dim=-1)  # [B,N]
        tbin = torch.clamp((t_idx.float() / alphas_cumprod.numel() * bins).long(), 0, bins - 1)
        for b in range(batch_size):
            eb = e_ids[b].view(-1).cpu().numpy()
            tb = int(tbin[b].item())
            for e in eb:
                heat[e, tb] += 1
    return heat


def evaluate_router_stability(model: ASTMEDiTSmall, alphas_cumprod: torch.Tensor, device: str, batch_size: int = 8) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.rand(batch_size, 3, model.img_size, model.img_size, device=device)
        t_idx = torch.randint(0, alphas_cumprod.numel(), (batch_size,), device=device)
        logsnr = index_to_logsnr(alphas_cumprod, t_idx)
        _ = model(x, logsnr)
        base = [blk._last_selection.argmax(dim=-1) for blk in model.blocks]  # list of [B,N]
        jitter = torch.clamp(x + 0.01 * torch.randn_like(x), 0.0, 1.0)
        _ = model(jitter, logsnr)
        jit = [blk._last_selection.argmax(dim=-1) for blk in model.blocks]
        diffs = [(b != j).float().mean().item() for b, j in zip(base, jit)]
        return float(np.mean(diffs))
