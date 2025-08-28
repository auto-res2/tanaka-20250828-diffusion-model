import os
from typing import Dict, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import (
    ensure_dir,
    device_and_dtype,
    PatternDataset,
    TrainConfig,
    build_model,
    q_sample,
    savefig_pdf,
)


def evaluate_model_checkpoint(models_dir: str, images_dir: str, ckpt_name: str = "rev_lora_student.pt") -> Dict[str, float]:
    ensure_dir(images_dir)
    ckpt_path = os.path.join(models_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return {"status": 0.0}

    device, base_dtype = device_and_dtype()

    payload = torch.load(ckpt_path, map_location=device)
    cfgd = payload.get("config", {})
    cfg = TrainConfig(**{**TrainConfig().__dict__, **cfgd})

    model = build_model(img_ch=3, cfg=cfg)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device=device, dtype=base_dtype)
    model.eval()

    # Simple held-out evaluation: predict epsilon and compute MSE on 64 samples
    dataset = PatternDataset(n=64, size=64, channels=3)
    imgs = torch.stack([dataset[i]["image"] for i in range(len(dataset))], dim=0).to(device=device, dtype=base_dtype)
    with torch.no_grad():
        t = torch.rand(imgs.size(0), device=device)
        noisy, eps = q_sample(imgs, t, 1e-4, 0.02)
        pred_eps = model(noisy)
        mse = torch.mean((pred_eps - eps) ** 2).item()

        # Visual grid
        sel = slice(0, min(8, imgs.size(0)))
        noisy_sel = noisy[sel].clamp(0, 1).cpu().numpy()
        denoised = (noisy[sel] - pred_eps[sel]).clamp(0, 1).cpu().numpy()
        fig, axes = plt.subplots(2, noisy_sel.shape[0], figsize=(12, 3))
        for i in range(noisy_sel.shape[0]):
            axes[0, i].imshow(np.transpose(noisy_sel[i], (1, 2, 0)))
            axes[0, i].axis('off')
            axes[1, i].imshow(np.transpose(denoised[i], (1, 2, 0)))
            axes[1, i].axis('off')
        plt.suptitle("Evaluation: Noisy (top) vs Denoised (bottom)")
        plt.tight_layout()
        savefig_pdf(os.path.join(images_dir, "eval_samples_revlora.pdf"))
        print(f"Saved: {os.path.join(images_dir, 'eval_samples_revlora.pdf')}")

    print(f"Evaluation MSE (epsilon prediction): {mse:.6f}")
    return {"mse": mse, "status": 1.0}
