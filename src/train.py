import os
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Utilities & Environment
# =============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig_pdf(path: str):
    d = os.path.dirname(path)
    if d:
        ensure_dir(d)
    plt.savefig(path, bbox_inches="tight")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float16 on CUDA, float32 on CPU (GroupNorm supports fp16 on GPU)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return device, dtype


def get_peak_gpu_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return float("nan")


# =============================
# LoRA-in-place for Conv2d
# =============================
class LoRAInjectedConv2d(nn.Module):
    """Conv2d wrapper with frozen base weights and trainable LoRA A,B.
    y = Conv(x; W) + scale * Conv(Conv(x; A), B)
    """
    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: float = 8.0):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.kernel_size = base.kernel_size
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels

        # Frozen base params
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        # LoRA params
        self.r = max(0, int(r))
        self.alpha = float(alpha)
        if self.r > 0:
            a_shape = (self.r, self.in_channels // self.groups, *self.kernel_size)
            b_shape = (self.out_channels, self.r, 1, 1)
            self.A = nn.Parameter(torch.zeros(a_shape))
            self.B = nn.Parameter(torch.zeros(b_shape))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            self.scaling = self.alpha / max(1, self.r)
        else:
            self.register_buffer("A", torch.zeros(0))
            self.register_buffer("B", torch.zeros(0))
            self.scaling = 0.0

    def forward(self, x):
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.r > 0:
            lora = F.conv2d(x, self.A, None, self.stride, self.padding, self.dilation, self.groups)
            lora = F.conv2d(lora, self.B, None, 1, 0, 1, 1)
            y = y + self.scaling * lora
        return y


def inject_lora_conv2d(module: nn.Module, r: int, alpha: Optional[float] = None,
                       include_first_last: bool = True, verbose: bool = False) -> int:
    """Recursively replace Conv2d with LoRA-injected version. Returns count replaced."""
    if alpha is None:
        alpha = float(r)
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            new = LoRAInjectedConv2d(child, r=r, alpha=alpha)
            setattr(module, name, new)
            count += 1
            if verbose:
                print(f"[LoRA] Replaced Conv2d at {name} with r={r}, alpha={alpha}")
        else:
            count += inject_lora_conv2d(child, r=r, alpha=alpha, include_first_last=include_first_last, verbose=verbose)
    return count


# =============================
# Sliced compute wrapper
# =============================
class SlicedWrapper(nn.Module):
    """Run a module over k height stripes to reduce peak activation footprint."""
    def __init__(self, mod: nn.Module, k: int = 1):
        super().__init__()
        self.mod = mod
        self.k = int(max(1, k))

    def forward(self, x):
        if self.k == 1:
            return self.mod(x)
        B, C, H, W = x.shape
        chunk_h = max(1, H // self.k)
        outs = []
        for i in range(self.k):
            s = slice(i * chunk_h, (i + 1) * chunk_h if i < self.k - 1 else H)
            outs.append(self.mod(x[:, :, s, :]))
        return torch.cat(outs, dim=2)


# =============================
# Reversible Block (custom autograd)
# =============================
class RevBlockFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f: nn.Module, g: nn.Module):
        x1, x2 = torch.chunk(x, 2, dim=1)
        with torch.no_grad():
            y1 = x1 + f(x2)
            y2 = x2 + g(y1)
        ctx.f = f
        ctx.g = g
        ctx.save_for_backward(y1, y2)
        return torch.cat([y1, y2], dim=1)

    @staticmethod
    def backward(ctx, dy):
        f, g = ctx.f, ctx.g
        y1, y2 = ctx.saved_tensors
        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        with torch.no_grad():
            dev = y1.device
            # Prefetch to device if offloaded
            if next(f.parameters(), None) is not None and next(f.parameters()).device != dev:
                f.to(dev)
            if next(g.parameters(), None) is not None and next(g.parameters()).device != dev:
                g.to(dev)
            x2 = y2 - g(y1)
            x1 = y1 - f(x2)
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        with torch.enable_grad():
            y1_hat = x1 + f(x2)
            y2_hat = x2 + g(y1_hat)
            torch.autograd.backward((y1_hat, y2_hat), (dy1, dy2))
        dx = torch.cat([x1.grad, x2.grad], dim=1)
        return dx, None, None


class RevBlock(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module, slice_k: int = 1):
        super().__init__()
        self.f = SlicedWrapper(f, k=slice_k)
        self.g = SlicedWrapper(g, k=slice_k)

    def forward(self, x):
        return RevBlockFn.apply(x, self.f, self.g)


# =============================
# Tiny RevUNet
# =============================
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvGNAct(ch, ch)
        self.conv2 = ConvGNAct(ch, ch)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class _FuseHalves(nn.Module):
    def __init__(self, inner: nn.Module, first_half: bool, ch: int):
        super().__init__()
        self.inner = inner
        self.first = first_half
        self.C = ch
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if self.first:
            x1 = self.inner(x1)
        else:
            x2 = self.inner(x2)
        return torch.cat([x1, x2], dim=1)


class OffloadScheduler:
    """Simple offloader: keeps last K modules on GPU, older on CPU."""
    def __init__(self, window_keep: int = 0, verbose: bool = False):
        self.window_keep = int(max(0, window_keep))
        self.verbose = verbose
        self.history: List[nn.Module] = []
        self.enabled = torch.cuda.is_available() and (self.window_keep > 0)

    def on_enter(self, module: nn.Module):
        if not self.enabled:
            return
        try:
            module.to("cuda", non_blocking=True)
        except Exception:
            pass

    def on_exit(self, module: nn.Module):
        if not self.enabled:
            return
        self.history.append(module)
        if len(self.history) > self.window_keep:
            idx = len(self.history) - self.window_keep - 1
            old = self.history[idx]
            try:
                old.to("cpu", non_blocking=True)
                if self.verbose:
                    print(f"[Offload] Moved module {old.__class__.__name__} to CPU")
            except Exception:
                pass


class TinyRevUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, channels=(32, 64), rev=True, slice_k=1, num_rev_per_stage=2):
        super().__init__()
        self.rev = rev
        self.slice_k = slice_k
        self.depth = len(channels)
        self.in_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        chs = channels
        for i in range(self.depth):
            stage = nn.ModuleList()
            ch = chs[i]
            if rev:
                assert ch % 2 == 0, "Channels must be even for reversible split"
                for _ in range(num_rev_per_stage):
                    f = ResidualBlock(ch // 2)
                    g = ResidualBlock(ch // 2)
                    stage.append(RevBlock(f, g, slice_k=slice_k))
            else:
                for _ in range(num_rev_per_stage * 2):
                    stage.append(ResidualBlock(ch))
            self.down_blocks.append(stage)
            if i < self.depth - 1:
                self.downs.append(Downsample(ch))

        self.mid = nn.ModuleList([ResidualBlock(chs[-1]) for _ in range(2)])

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.depth)):
            if i < self.depth - 1:
                self.ups.append(Upsample(chs[i]))
            stage = nn.ModuleList()
            ch = chs[i]
            if rev:
                for _ in range(num_rev_per_stage):
                    f = ResidualBlock(ch // 2)
                    g = ResidualBlock(ch // 2)
                    stage.append(RevBlock(f, g, slice_k=slice_k))
            else:
                for _ in range(num_rev_per_stage * 2):
                    stage.append(ResidualBlock(ch))
            self.up_blocks.append(stage)

        self.out_conv = nn.Conv2d(channels[0], out_ch, 3, padding=1)

    def forward(self, x, offloader: Optional['OffloadScheduler'] = None):
        feats = []
        h = self.in_conv(x)
        for si, stage in enumerate(self.down_blocks):
            for block in stage:
                if offloader is not None:
                    offloader.on_enter(block)
                h = block(h)
                if offloader is not None:
                    offloader.on_exit(block)
            feats.append(h)
            if si < len(self.downs):
                h = self.downs[si](h)
        for block in self.mid:
            if offloader is not None:
                offloader.on_enter(block)
            h = block(h)
            if offloader is not None:
                offloader.on_exit(block)
        for si, stage in enumerate(self.up_blocks):
            if si > 0:
                h = self.ups[si - 1](h)
            skip = feats[-(si + 1)]
            h = h + skip
            for block in stage:
                if offloader is not None:
                    offloader.on_enter(block)
                h = block(h)
                if offloader is not None:
                    offloader.on_exit(block)
        y = self.out_conv(h)
        return y


# =============================
# Synthetic Data
# =============================
class PatternDataset(Dataset):
    def __init__(self, n: int = 512, size: int = 64, patterns: Optional[List[str]] = None, channels: int = 3):
        super().__init__()
        self.n = n
        self.size = size
        self.channels = channels
        self.patterns = patterns or ["stripes", "checker", "blobs"]

    def __len__(self):
        return self.n

    def _stripes(self):
        H = W = self.size
        img = np.zeros((H, W), dtype=np.float32)
        stripe_w = random.randint(2, 6)
        for i in range(0, H, stripe_w * 2):
            img[i:i + stripe_w, :] = 1.0
        return img

    def _checker(self):
        H = W = self.size
        img = np.indices((H, W)).sum(axis=0) % 2
        img = img.astype(np.float32)
        return img

    def _blobs(self):
        H = W = self.size
        img = np.zeros((H, W), dtype=np.float32)
        for _ in range(random.randint(3, 7)):
            cx, cy = random.randint(0, W - 1), random.randint(0, H - 1)
            r = random.randint(3, max(4, self.size // 6))
            y, x = np.ogrid[:H, :W]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
            img[mask] = 1.0
        return img

    def __getitem__(self, idx):
        p = random.choice(self.patterns)
        if p == "stripes":
            base = self._stripes()
        elif p == "checker":
            base = self._checker()
        else:
            base = self._blobs()
        base = base[None, :, :]
        base = np.repeat(base, self.channels, axis=0)
        base = torch.from_numpy(base)
        base = base + 0.05 * torch.randn_like(base)
        base = base.clamp(0.0, 1.0)
        return {"image": base}


# =============================
# Diffusion-like utilities
# =============================
@dataclass
class TrainConfig:
    lr: float = 1e-3
    steps: int = 200
    batch_size: int = 4
    img_size: int = 64
    lora_rank: int = 8
    slice_k: int = 1
    rev: bool = True
    window_keep: int = 0
    noise_beta_min: float = 1e-4
    noise_beta_max: float = 0.02
    teacher_kl_prob: float = 0.1
    teacher_kl_w: float = 0.1


def linear_beta_schedule(t: torch.Tensor, beta_min: float, beta_max: float):
    return beta_min + t * (beta_max - beta_min)


def q_sample(x0: torch.Tensor, t: torch.Tensor, beta_min: float, beta_max: float):
    eps = torch.randn_like(x0)
    beta_t = linear_beta_schedule(t, beta_min, beta_max).view(-1, 1, 1, 1)
    noisy = x0 + (beta_t.sqrt()) * eps
    return noisy, eps


def build_model(img_ch: int, cfg: TrainConfig) -> TinyRevUNet:
    model = TinyRevUNet(in_ch=img_ch, out_ch=img_ch, channels=(32, 32), rev=cfg.rev, slice_k=cfg.slice_k, num_rev_per_stage=2)
    if cfg.lora_rank > 0:
        inject_lora_conv2d(model, r=cfg.lora_rank, alpha=cfg.lora_rank, include_first_last=True, verbose=False)
        for n, p in model.named_parameters():
            if p.requires_grad and ("A" in n or "B" in n):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    return model


def diffusion_step(student: nn.Module, batch: Dict[str, torch.Tensor],
                   beta_min: float, beta_max: float,
                   teacher: Optional[nn.Module] = None,
                   kl_prob: float = 0.0, kl_w: float = 0.1,
                   offloader: Optional[OffloadScheduler] = None,
                   autocast_dtype: Optional[torch.dtype] = None):
    x0 = batch["image"]
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)
    noisy, eps = q_sample(x0, t, beta_min, beta_max)
    amp_ctx = torch.amp.autocast(device_type='cuda', enabled=(autocast_dtype is not None), dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    with amp_ctx:
        pred = student(noisy, offloader=offloader)
        mse = F.mse_loss(pred, eps)
        loss = mse
        kl_val = torch.tensor(0.0, device=x0.device)
        if teacher is not None and random.random() < kl_prob:
            with torch.no_grad():
                teacher_pred = teacher(noisy)
            kl_val = F.mse_loss(pred, teacher_pred)
            loss = loss + kl_w * kl_val
    logs = {"mse": float(mse.detach().cpu().item()), "kl": float(kl_val.detach().cpu().item())}
    return loss, logs


# =============================
# Experiments
# =============================
@dataclass
class Variant:
    name: str
    rev: bool
    lora_rank: int
    slice_k: int
    window_keep: int


def run_experiment_1(images_dir: str) -> List[Dict[str, float]]:
    print("=== Experiment 1: Memory/runtime validation with component toggles ===")
    seed_everything(123)
    device, base_dtype = device_and_dtype()
    autocast_dtype = torch.float16 if (device.type == 'cuda') else None

    variants = [
        Variant(name="baseline", rev=False, lora_rank=0, slice_k=1, window_keep=0),
        Variant(name="rev_only", rev=True, lora_rank=0, slice_k=1, window_keep=0),
        Variant(name="rev_lora_r8", rev=True, lora_rank=8, slice_k=1, window_keep=0),
        Variant(name="rev_lora_r8_k2", rev=True, lora_rank=8, slice_k=2, window_keep=0),
        Variant(name="rev_lora_r8_k2_offload2", rev=True, lora_rank=8, slice_k=2, window_keep=2),
    ]

    dataset = PatternDataset(n=64, size=64, channels=3)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch_iter = iter(loader)
    batch = next(batch_iter)
    for k in batch:
        batch[k] = batch[k].to(device=device, dtype=base_dtype)

    results = []
    for v in variants:
        print(f"\n[Variant] {v.name} -> {asdict(v)}")
        model = build_model(img_ch=3, cfg=TrainConfig(rev=v.rev, lora_rank=v.lora_rank, slice_k=v.slice_k))
        model.to(device=device, dtype=base_dtype)
        offloader = OffloadScheduler(window_keep=v.window_keep, verbose=False)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        step_times, losses = [], []
        n_warm, n_meas = 3, 10
        for step in range(n_warm + n_meas):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)
            for k in batch:
                batch[k] = batch[k].to(device=device, dtype=base_dtype)

            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss, logs = diffusion_step(model, batch,
                                        beta_min=1e-4, beta_max=0.02,
                                        teacher=None,
                                        kl_prob=0.0, kl_w=0.0,
                                        offloader=offloader,
                                        autocast_dtype=autocast_dtype)
            loss.backward()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if step >= n_warm:
                step_times.append((t1 - t0) * 1000.0)
                losses.append(float(loss.detach().cpu().item()))
        peak_gb = get_peak_gpu_gb()
        print(f"Variant {v.name}: mean step time = {np.mean(step_times):.2f} ms, peak GPU mem = {peak_gb:.3f} GB")
        results.append({"name": v.name, "time_ms": np.mean(step_times), "peak_gb": peak_gb, "loss": np.mean(losses)})

    names = [r["name"] for r in results]
    times = [r["time_ms"] for r in results]
    mems = [r["peak_gb"] for r in results]

    plt.figure(figsize=(6, 3))
    sns.barplot(x=names, y=times, palette="Blues_d")
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Step time (ms)")
    plt.title("Per-step runtime across variants")
    plt.tight_layout()
    savefig_pdf(os.path.join(images_dir, "inference_latency_revlora.pdf"))
    print(f"Saved: {os.path.join(images_dir, 'inference_latency_revlora.pdf')}")

    plt.figure(figsize=(6, 3))
    sns.barplot(x=names, y=mems, palette="Greens_d")
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Peak GPU memory (GB)")
    plt.title("Peak memory across variants")
    plt.tight_layout()
    savefig_pdf(os.path.join(images_dir, "peak_memory_revlora.pdf"))
    print(f"Saved: {os.path.join(images_dir, 'peak_memory_revlora.pdf')}")

    return results


@dataclass
class Exp2Config:
    steps: int = 200
    batch_size: int = 8
    img_size: int = 64
    lora_rank: int = 8
    slice_k: int = 2
    rev: bool = True
    kl_prob: float = 0.1
    kl_w: float = 0.1


def run_experiment_2(images_dir: str, models_dir: str) -> Dict[str, float]:
    print("\n=== Experiment 2: Quality parity on synthetic diffusion task ===")
    seed_everything(7)
    device, base_dtype = device_and_dtype()
    autocast_dtype = torch.float16 if device.type == 'cuda' else None

    cfg = Exp2Config(steps=200, batch_size=8, img_size=64, lora_rank=8, slice_k=2, rev=True, kl_prob=0.1, kl_w=0.1)

    dataset = PatternDataset(n=512, size=cfg.img_size, channels=3)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    student = build_model(img_ch=3, cfg=TrainConfig(rev=cfg.rev, lora_rank=cfg.lora_rank, slice_k=cfg.slice_k))
    teacher = build_model(img_ch=3, cfg=TrainConfig(rev=cfg.rev, lora_rank=0, slice_k=1))
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.to(device=device, dtype=base_dtype)
    teacher.to(device=device, dtype=base_dtype)

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)

    losses, mses, kls, grad_norms = [], [], [], []
    nans = 0

    for step, batch in enumerate(loader):
        if step >= cfg.steps:
            break
        for k in batch:
            batch[k] = batch[k].to(device=device, dtype=base_dtype)
        opt.zero_grad(set_to_none=True)
        loss, logs = diffusion_step(student, batch,
                                    beta_min=1e-4, beta_max=0.02,
                                    teacher=teacher,
                                    kl_prob=cfg.kl_prob, kl_w=cfg.kl_w,
                                    offloader=None,
                                    autocast_dtype=autocast_dtype)
        if not torch.isfinite(loss):
            nans += 1
            print(f"NaN at step {step}")
            continue
        loss.backward()
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        grad_norms.append(math.sqrt(total_norm))
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        mses.append(logs["mse"])
        kls.append(logs.get("kl", 0.0))
        if step % 20 == 0:
            print(f"Step {step:04d}: loss={losses[-1]:.4f} | mse={mses[-1]:.4f} | kl={kls[-1]:.4f} | grad_norm={grad_norms[-1]:.3f}")

    print(f"Finished: steps={cfg.steps}, NaN steps={nans}")

    x = np.arange(len(losses))
    plt.figure(figsize=(6, 3))
    plt.plot(x, losses, label="loss")
    plt.plot(x, mses, label="mse")
    plt.plot(x, kls, label="kl")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training loss components (RevLoRA)")
    plt.tight_layout()
    savefig_pdf(os.path.join(images_dir, "training_loss_revlora.pdf"))
    print(f"Saved: {os.path.join(images_dir, 'training_loss_revlora.pdf')}")

    plt.figure(figsize=(6, 3))
    plt.plot(x, grad_norms)
    plt.xlabel("Step")
    plt.ylabel("Grad norm (LoRA)")
    plt.title("Gradient norm over training")
    plt.tight_layout()
    savefig_pdf(os.path.join(images_dir, "gradient_norm_revlora.pdf"))
    print(f"Saved: {os.path.join(images_dir, 'gradient_norm_revlora.pdf')}")

    # Qualitative samples
    with torch.no_grad():
        batch = next(iter(loader))
        imgs = batch["image"].to(device=device, dtype=base_dtype)[:8]
        noisy, eps = q_sample(imgs, torch.rand(imgs.size(0), device=imgs.device), 1e-4, 0.02)
        pred_eps = student(noisy)
        denoised = noisy - pred_eps
        denoised = denoised.clamp(0, 1).cpu().numpy()
        fig, axes = plt.subplots(2, 8, figsize=(12, 3))
        for i in range(8):
            axes[0, i].imshow(np.transpose(noisy[i].clamp(0, 1).cpu().numpy(), (1, 2, 0)))
            axes[0, i].axis('off')
            axes[1, i].imshow(np.transpose(denoised[i], (1, 2, 0)))
            axes[1, i].axis('off')
        plt.suptitle("Top: Noisy | Bottom: Denoised (RevLoRA)")
        plt.tight_layout()
        savefig_pdf(os.path.join(images_dir, "samples_revlora.pdf"))
        print(f"Saved: {os.path.join(images_dir, 'samples_revlora.pdf')}")

    # Save checkpoint (LoRA weights only)
    ensure_dir(models_dir)
    ckpt_path = os.path.join(models_dir, "rev_lora_student.pt")
    torch.save({"state_dict": student.state_dict(), "config": asdict(TrainConfig(lora_rank=cfg.lora_rank, slice_k=cfg.slice_k, rev=cfg.rev))}, ckpt_path)
    print(f"Saved model checkpoint: {ckpt_path}")

    return {
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "mse_mean": float(np.mean(mses)),
        "kl_mean": float(np.mean(kls)),
        "grad_norm_mean": float(np.mean(grad_norms)),
    }


def gradcheck_revblock_small() -> bool:
    torch.manual_seed(0)
    ch = 32
    f = ResidualBlock(ch // 2)
    g = ResidualBlock(ch // 2)
    blk = RevBlock(f, g, slice_k=2)
    blk.eval()
    blk.double()
    x = torch.randn(1, ch, 8, 8, dtype=torch.double, requires_grad=True)
    def fn(z):
        return blk(z).sum()
    ok = torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)
    return ok


def run_experiment_3(images_dir: str) -> List[Dict[str, float]]:
    print("\n=== Experiment 3: Component attribution, robustness, grad correctness ===")
    seed_everything(2024)
    ok = gradcheck_revblock_small()
    print(f"Gradcheck (RevBlock, toy): {ok}")

    device, base_dtype = device_and_dtype()
    autocast_dtype = torch.float16 if device.type == 'cuda' else None

    ranks = [4, 16]
    ks = [1, 2]
    ws = [0, 2]

    grid_results = []
    for r in ranks:
        for k in ks:
            for w in ws:
                cfg = TrainConfig(lr=1e-3, steps=60, batch_size=4, img_size=64, lora_rank=r, slice_k=k, rev=True, window_keep=w)
                dataset = PatternDataset(n=120, size=cfg.img_size, channels=3)
                loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

                model = build_model(img_ch=3, cfg=cfg)
                model.to(device=device, dtype=base_dtype)
                offloader = OffloadScheduler(window_keep=w)
                params = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=cfg.lr)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                losses, t_accum = [], []
                for step, batch in enumerate(loader):
                    if step >= cfg.steps:
                        break
                    for key in batch:
                        batch[key] = batch[key].to(device=device, dtype=base_dtype)
                    t0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    loss, _ = diffusion_step(model, batch,
                                              beta_min=1e-4, beta_max=0.02,
                                              teacher=None,
                                              kl_prob=0.0, kl_w=0.0,
                                              offloader=offloader,
                                              autocast_dtype=autocast_dtype)
                    loss.backward()
                    opt.step()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    t_accum.append((t1 - t0) * 1000.0)
                    losses.append(float(loss.detach().cpu().item()))
                peak_gb = get_peak_gpu_gb()
                res = {"r": r, "k": k, "w": w,
                       "time_ms": float(np.mean(t_accum)),
                       "peak_gb": peak_gb,
                       "loss_mean": float(np.mean(losses)),
                       "loss_std": float(np.std(losses))}
                print(f"Config r={r}, k={k}, w={w} -> time {res['time_ms']:.2f} ms, peak {res['peak_gb']:.3f} GB, loss {res['loss_mean']:.4f}")
                grid_results.append(res)

    # Memory vs slicing factor per rank
    plt.figure(figsize=(6, 3))
    for r in ranks:
        mem_by_k = []
        for k in ks:
            subset = [v for v in grid_results if v["r"] == r and v["k"] == k]
            mem_by_k.append(np.nanmean([v["peak_gb"] for v in subset]))
        plt.plot(ks, mem_by_k, marker='o', label=f"rank={r}")
    plt.xlabel("slice factor k")
    plt.ylabel("Peak GPU memory (GB)")
    plt.title("Memory vs slicing factor")
    plt.legend()
    plt.tight_layout()
    savefig_pdf(os.path.join(images_dir, "peak_memory_tokens_revlora.pdf"))
    print(f"Saved: {os.path.join(images_dir, 'peak_memory_tokens_revlora.pdf')}")

    # Time vs offload window
    plt.figure(figsize=(6, 3))
    for r in ranks:
        for k in ks:
            subset = [v for v in grid_results if v["r"] == r and v["k"] == k]
            xs = sorted(set(v["w"] for v in subset))
            ys = [np.mean([vv["time_ms"] for vv in subset if vv["w"] == x]) for x in xs]
            plt.plot(xs, ys, marker='s', label=f"r={r},k={k}")
    plt.xlabel("offload window (w)")
    plt.ylabel("Step time (ms)")
    plt.title("Runtime vs offload window")
    plt.legend()
    plt.tight_layout()
    savefig_pdf(os.path.join(images_dir, "inference_latency_tokens_revlora.pdf"))
    print(f"Saved: {os.path.join(images_dir, 'inference_latency_tokens_revlora.pdf')}")

    return grid_results


# =============================
# Quick test orchestrator
# =============================

def quick_test(images_dir: str, models_dir: str) -> Dict[str, object]:
    print("\n>>> Running quick_test() ...")
    ensure_dir(images_dir)
    ensure_dir(models_dir)
    res1 = run_experiment_1(images_dir=images_dir)
    res2 = run_experiment_2(images_dir=images_dir, models_dir=models_dir)
    res3 = run_experiment_3(images_dir=images_dir)
    print("\nTest complete. Key summaries:")
    print("Exp1 variants:")
    for r in res1:
        print(r)
    print("\nExp2 summary:")
    print(res2)
    print("\nExp3 grid sample (first 3):")
    for r in res3[:3]:
        print(r)
    return {"exp1": res1, "exp2": res2, "exp3": res3}
