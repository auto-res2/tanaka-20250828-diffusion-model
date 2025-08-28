import os
import math
import time
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import pynvml
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False

try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

from .preprocess import (
    ensure_dir, set_seed, get_device,
    make_linear_schedule, create_dataloaders, make_canonical_set,
    SinusoidalTimeEmbedding, GlobalCondMLP, DiffusionSchedule,
    q_sample, ddim_step, ddpm_step, dpmpp_heun_step,
)
from .train import (
    BaselineUNet, R2DiffUNet, TokenBackbone, R2DiffToken, train_baseline, train_r2diff,
)


plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['pdf.fonttype'] = 42
sns.set_context('paper')


def save_image_grid(images: torch.Tensor, filename: str, nrow: int = 8):
    images = images.detach().cpu().clamp(-1,1)
    images = (images + 1.0) / 2.0
    B, C, H, W = images.shape
    ncol = nrow
    nrow = int(np.ceil(B / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*1.2, nrow*1.2))
    axes = np.array(axes).reshape(nrow, ncol)
    idx = 0
    for r in range(nrow):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis('off')
            if idx < B:
                img = images[idx,0].numpy()
                ax.imshow(img, cmap='gray', vmin=0.0, vmax=1.0)
            idx += 1
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def measure_run(fn, *args, warmup=1, repeat=3) -> Dict[str, float]:
    device = get_device()
    for _ in range(warmup):
        _ = fn(*args)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    times = []
    energies = []
    peak_mem_mb = 0.0
    if _HAS_PYNVML and device.type == 'cuda':
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            has_energy = True
        except Exception:
            has_energy = False
    else:
        has_energy = False
    for _ in range(repeat):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if has_energy:
            try:
                e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            except Exception:
                e0 = None
        _ = fn(*args)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        if has_energy and e0 is not None:
            try:
                e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                energies.append(max(0.0, (e1 - e0) / 1000.0))
            except Exception:
                pass
        times.append(t1 - t0)
        if device.type == 'cuda':
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / (1024**2))
    if _HAS_PYNVML and device.type == 'cuda':
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return {
        'latency_ms_mean': float(1000 * np.mean(times)),
        'latency_ms_std': float(1000 * np.std(times)),
        'energy_j_mean': float(np.mean(energies)) if len(energies) > 0 else float('nan'),
        'peak_mem_mb': peak_mem_mb
    }


def plot_curve(values: List[float], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(5,3))
    sns.lineplot(x=list(range(len(values))), y=values)
    plt.title(title)
    plt.xlabel('iteration')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_bars(labels: List[str], values: List[float], title: str, ylabel: str, filename: str):
    plt.figure(figsize=(5,3))
    sns.barplot(x=labels, y=values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_pdf(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], filename: str):
    if not _HAS_SKLEARN:
        print("sklearn not installed; skipping confusion matrix plot.")
        return
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# ------------------------------
# Classifier for confusion matrix
# ------------------------------

class TinyClassifier(torch.nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.AvgPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(32*8*8, 64), torch.nn.ReLU(), torch.nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(ds, epochs=2, lr=1e-3, device=None) -> TinyClassifier:
    device = device or get_device()
    clf = TinyClassifier().to(device)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(epochs):
        for x0, _, cls, _ in dl:
            x0 = x0.to(device); y = cls.to(device)
            logits = clf(x0)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    return clf


def evaluate_confusion(clf: TinyClassifier, imgs: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        logits = clf(imgs)
        preds = logits.argmax(dim=-1).cpu().numpy()
        y_true = labels.cpu().numpy()
    return y_true, preds


# ------------------------------
# Sampling wrappers
# ------------------------------

def sample_images_baseline(model: BaselineUNet, cond_mlp: GlobalCondMLP, t_embed: SinusoidalTimeEmbedding,
                           sched: DiffusionSchedule, cond_vecs: torch.Tensor,
                           sampler='ddim', steps=25, device=None) -> torch.Tensor:
    device = device or get_device()
    model.eval(); cond_mlp.eval(); t_embed.eval()
    B = cond_vecs.size(0)
    H = W = 32
    x = torch.randn(B, 1, H, W, device=device)
    Ts = sched.betas.shape[0]
    t_indices = torch.linspace(Ts-1, 0, steps, device=device).long().tolist()

    def eps_fn(x_t, t_idx, cond_emb):
        return model(x_t, cond_emb)

    for i, t_idx in enumerate(t_indices):
        t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
        cond_emb = cond_mlp(cond_vecs, t_embed(t_tensor))
        eps = eps_fn(x, t_idx, cond_emb)
        t_prev = t_indices[i+1] if i+1 < len(t_indices) else -1
        if sampler == 'ddim':
            x = ddim_step(x, eps, t_idx, t_prev, sched)
        elif sampler == 'ddpm':
            x = ddpm_step(x, eps, t_idx, sched)
        elif sampler == 'dpmpp':
            x = dpmpp_heun_step(x, lambda xx, tt, cc: eps_fn(xx, tt, cc), t_idx, t_prev, sched, cond_emb)
        elif sampler == 'lcm':
            x = ddim_step(x, eps, t_idx, t_prev, sched)
        else:
            raise ValueError(f"Unknown sampler {sampler}")
    return x.clamp(-1,1)


def sample_images_r2(model_r2: R2DiffUNet, cond_mlp: GlobalCondMLP, t_embed: SinusoidalTimeEmbedding,
                     sched: DiffusionSchedule, cond_vecs: torch.Tensor,
                     sampler='ddim', steps=25, device=None) -> torch.Tensor:
    device = device or get_device()
    model_r2.eval(); cond_mlp.eval(); t_embed.eval()
    B = cond_vecs.size(0)
    H = W = 32
    x = torch.randn(B, 1, H, W, device=device)
    Ts = sched.betas.shape[0]
    t_indices = torch.linspace(Ts-1, 0, steps, device=device).long().tolist()

    t0 = torch.full((B,), t_indices[0], device=device, dtype=torch.long)
    cond_emb = cond_mlp(cond_vecs, t_embed(t0))
    eps, feats_q, scales, zps = model_r2.forward_first(x, cond_emb)
    t_prev = t_indices[1] if len(t_indices) > 1 else -1
    if sampler == 'ddim':
        x = ddim_step(x, eps, t_indices[0], t_prev, sched)
    elif sampler == 'ddpm':
        x = ddpm_step(x, eps, t_indices[0], sched)
    elif sampler == 'dpmpp':
        x = dpmpp_heun_step(
            x,
            lambda xx, tt, cc: model_r2.decoder(
                xx,
                [model_r2.encoder_recur._dequant(fq, sc.to(device), zp.to(device)) for fq, sc, zp in zip(feats_q, scales, zps)],
                cc
            ),
            t_indices[0], t_prev, sched, cond_emb
        )
    elif sampler == 'lcm':
        x = ddim_step(x, eps, t_indices[0], t_prev, sched)

    for i in range(1, len(t_indices)):
        t_idx = t_indices[i]
        t_prev = t_indices[i+1] if i+1 < len(t_indices) else -1
        t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
        cond_emb = cond_mlp(cond_vecs, t_embed(t_tensor))
        eps, feats_q, scales, zps = model_r2.forward_recurrent(x, cond_emb, feats_q, scales, zps)
        if sampler == 'ddim':
            x = ddim_step(x, eps, t_idx, t_prev, sched)
        elif sampler == 'ddpm':
            x = ddpm_step(x, eps, t_idx, sched)
        elif sampler == 'dpmpp':
            x = dpmpp_heun_step(
                x,
                lambda xx, tt, cc: model_r2.decoder(
                    xx,
                    [model_r2.encoder_recur._dequant(fq, sc.to(device), zp.to(device)) for fq, sc, zp in zip(feats_q, scales, zps)],
                    cc
                ),
                t_idx, t_prev, sched, cond_emb
            )
        elif sampler == 'lcm':
            x = ddim_step(x, eps, t_idx, t_prev, sched)
    return x.clamp(-1,1)


# ------------------------------
# Experiments
# ------------------------------

def experiment1(output_dir: str, quick: bool = True):
    ensure_dir(output_dir)
    set_seed(7)
    device = get_device()
    img_size = 32
    T = 30 if quick else 50
    sched = make_linear_schedule(T=T, device=device)

    n_train = 512 if quick else 2048
    n_val = 128 if quick else 512
    ds_train, ds_val, dl_train, dl_val = create_dataloaders(n_train, n_val, img_size, batch_size=64 if quick else 128)
    canonical_set = make_canonical_set(n=64, img_size=img_size)

    t_embed = SinusoidalTimeEmbedding(dim=64)
    cond_mlp = GlobalCondMLP(cond_dim=8, t_dim=64, out_dim=128)

    baseline = BaselineUNet(img_ch=1, base_ch=16, d_embed=128)
    print("Training baseline UNet (toy)...")
    tr_b, val_b = train_baseline(baseline, dl_train, dl_val, sched, cond_mlp, t_embed, epochs=2 if quick else 4, lr=1e-3, device=device)
    plot_curve(tr_b, 'Baseline training loss', 'MSE', os.path.join(output_dir, 'training_loss_baseline.pdf'))

    r2 = R2DiffUNet(img_ch=1, base_ch=16, d_embed=128, delta_depth=1, quant_mode='int8')
    print("Training R2Diff (toy) with teacher forcing and int8 state...")
    logs_r2 = train_r2diff(r2, baseline, dl_train, dl_val, sched, cond_mlp, t_embed,
                           epochs=2 if quick else 4, lr=1e-3, device=device, K_range=(2,3), distill_w=0.25, gate_reg=1e-3)
    plot_curve(logs_r2.train_losses, 'R2Diff training loss', 'MSE', os.path.join(output_dir, 'training_loss_r2diff.pdf'))
    plot_curve(logs_r2.gate_means, 'R2Diff mean gating over iterations', 'mean(γ)', os.path.join(output_dir, 'gating_values.pdf'))

    cond_vecs = torch.stack([c for _, c, _, _ in canonical_set]).to(device)
    labels = torch.tensor([cl for _,_,cl,_ in canonical_set], device=device)

    def gen_baseline():
        return sample_images_baseline(baseline, cond_mlp, t_embed, sched, cond_vecs, sampler='ddim', steps=15 if quick else 25, device=device)

    def gen_r2():
        return sample_images_r2(r2, cond_mlp, t_embed, sched, cond_vecs, sampler='ddim', steps=15 if quick else 25, device=device)

    print("Measuring baseline sampling...")
    m_base = measure_run(gen_baseline)
    print("Baseline sampling stats:", m_base)
    print("Measuring R2Diff sampling...")
    m_r2 = measure_run(gen_r2)
    print("R2Diff sampling stats:", m_r2)

    plot_bars(['Baseline','R2Diff'], [m_base['latency_ms_mean'], m_r2['latency_ms_mean']], 'Sampling latency (lower is better)', 'ms', os.path.join(output_dir, 'inference_latency.pdf'))
    plot_bars(['Baseline','R2Diff'], [m_base['peak_mem_mb'], m_r2['peak_mem_mb']], 'Peak memory (lower is better)', 'MB', os.path.join(output_dir, 'peak_memory.pdf'))

    x0_canon = torch.stack([x for x,_,_,_ in canonical_set]).to(device)
    with torch.no_grad():
        imgs_b = gen_baseline()
        imgs_r2 = gen_r2()
        save_image_grid(imgs_b, os.path.join(output_dir, 'samples_baseline.pdf'))
        save_image_grid(imgs_r2, os.path.join(output_dir, 'samples_r2diff.pdf'))
        mse_b = F.mse_loss(imgs_b, x0_canon).item()
        mse_r2 = F.mse_loss(imgs_r2, x0_canon).item()
        psnr_b = -10.0 * math.log10(mse_b + 1e-12)
        psnr_r2 = -10.0 * math.log10(mse_r2 + 1e-12)
    print(f"PSNR (dB): Baseline={psnr_b:.2f}, R2Diff={psnr_r2:.2f}")

    print("Training tiny classifier on real patterns for confusion matrix...")
    clf = train_classifier(ds_train if quick else ds_train, epochs=2 if quick else 3, lr=1e-3, device=device)
    y_true_b, y_pred_b = evaluate_confusion(clf, imgs_b, labels)
    y_true_r, y_pred_r = evaluate_confusion(clf, imgs_r2, labels)
    plot_confusion_matrix_pdf(y_true_b, y_pred_b, ['stripes','checker','circle','blobs'], os.path.join(output_dir, 'confusion_matrix_baseline.pdf'))
    plot_confusion_matrix_pdf(y_true_r, y_pred_r, ['stripes','checker','circle','blobs'], os.path.join(output_dir, 'confusion_matrix_r2diff.pdf'))

    steps_list = [5, 10, 15] if quick else [5, 8, 15, 25]
    mse_steps_base, mse_steps_r2 = [], []
    for s in steps_list:
        imgs_b_s = sample_images_baseline(baseline, cond_mlp, t_embed, sched, cond_vecs, sampler='ddim', steps=s, device=device)
        imgs_r2_s = sample_images_r2(r2, cond_mlp, t_embed, sched, cond_vecs, sampler='ddim', steps=s, device=device)
        mse_steps_base.append(F.mse_loss(imgs_b_s, x0_canon).item())
        mse_steps_r2.append(F.mse_loss(imgs_r2_s, x0_canon).item())
    plt.figure(figsize=(5,3))
    plt.plot(steps_list, mse_steps_base, marker='o', label='Baseline')
    plt.plot(steps_list, mse_steps_r2, marker='o', label='R2Diff')
    plt.xlabel('Sampling steps'); plt.ylabel('MSE (lower better)'); plt.title('Robustness vs steps')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_steps.pdf'), bbox_inches='tight')
    plt.close()

    print("Experiment 1 completed. Outputs saved in:", output_dir)
    return {
        'baseline_metrics': m_base,
        'r2diff_metrics': m_r2,
        'psnr_baseline': psnr_b,
        'psnr_r2diff': psnr_r2
    }


def experiment2(output_dir: str, quick: bool = True):
    ensure_dir(output_dir)
    set_seed(11)
    device = get_device()
    img_size = 32
    T = 30 if quick else 50
    sched = make_linear_schedule(T=T, device=device)

    ds_train, ds_val, dl_train, dl_val = create_dataloaders(512 if quick else 1024, 128 if quick else 256, img_size, batch_size=64)
    canonical_set = make_canonical_set(n=48, img_size=img_size)
    cond_vecs = torch.stack([c for _, c, _, _ in canonical_set]).to(device)
    x0_canon = torch.stack([x for x,_,_,_ in canonical_set]).to(device)
    t_embed = SinusoidalTimeEmbedding(dim=64)
    cond_mlp = GlobalCondMLP(cond_dim=8, t_dim=64, out_dim=128)

    base = BaselineUNet(1, 16, 128).to(device)
    train_baseline(base, dl_train, dl_val, sched, cond_mlp, t_embed, epochs=2 if quick else 3, lr=1e-3, device=device)
    r2cnn = R2DiffUNet(1, 16, 128, delta_depth=1, quant_mode='int8').to(device)
    train_r2diff(r2cnn, base, dl_train, dl_val, sched, cond_mlp, t_embed, epochs=2 if quick else 3, lr=1e-3, device=device)

    base_tok = TokenBackbone(img_size=img_size, patch=4, dim=64, d_embed=128).to(device)
    opt = torch.optim.Adam(base_tok.parameters(), lr=1e-3)
    for _ in range(2 if quick else 3):
        for x0, cond, _, _ in dl_train:
            x0 = x0.to(device); cond = cond.to(device)
            B = x0.size(0)
            t = torch.randint(low=0, high=sched.betas.shape[0], size=(B,), device=device)
            x_t, eps = q_sample(x0, t, sched)
            cond_emb = cond_mlp(cond, t_embed(t))
            pred = base_tok(x_t, cond_emb)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()

    r2tok = R2DiffToken(img_size=img_size, patch=4, dim=64, d_embed=128, delta_depth=1, quant_mode='int8').to(device)
    opt2 = torch.optim.Adam(r2tok.parameters(), lr=1e-3)
    for _ in range(2 if quick else 3):
        for x0, cond, _, _ in dl_train:
            x0 = x0.to(device); cond = cond.to(device)
            B = x0.size(0)
            t0 = torch.full((B,), T-1, device=device, dtype=torch.long)
            x_t, eps_gt = q_sample(x0, t0, sched)
            cond_emb = cond_mlp(cond, t_embed(t0))
            eps1, q, sc, zp = r2tok.forward_first(x_t, cond_emb)
            with torch.no_grad():
                eps_teacher = base_tok(x_t, cond_emb)
            loss = F.mse_loss(eps1, eps_gt) + 0.25 * F.mse_loss(eps1, eps_teacher)
            opt2.zero_grad(); loss.backward(); opt2.step()

    samplers = ['ddpm','ddim','dpmpp','lcm']
    steps_map = {'ddpm': [30 if quick else 50], 'ddim':[8,15] if quick else [5,8,15,25], 'dpmpp':[15], 'lcm':[8]}

    results = []
    for sname in samplers:
        for steps in steps_map[sname]:
            imgs_b = sample_images_baseline(base, cond_mlp, t_embed, sched, cond_vecs, sampler=sname, steps=steps, device=device)
            imgs_r = sample_images_r2(r2cnn, cond_mlp, t_embed, sched, cond_vecs, sampler=sname, steps=steps, device=device)
            mse_b = F.mse_loss(imgs_b, x0_canon).item()
            mse_r = F.mse_loss(imgs_r, x0_canon).item()
            psnr_b = -10.0 * math.log10(mse_b + 1e-12)
            psnr_r = -10.0 * math.log10(mse_r + 1e-12)
            results.append({'sampler': sname, 'steps': steps, 'psnr_baseline': psnr_b, 'psnr_r2': psnr_r})
            print(f"[Sampler={sname}, steps={steps}] PSNR: Baseline={psnr_b:.2f} dB, R2Diff={psnr_r:.2f} dB")

    labels = [f"{r['sampler']}-{r['steps']}" for r in results]
    psnr_bars = [r['psnr_baseline'] for r in results]
    psnr_bars_r2 = [r['psnr_r2'] for r in results]
    plt.figure(figsize=(7,3))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, psnr_bars, width, label='Baseline')
    plt.bar(x + width/2, psnr_bars_r2, width, label='R2Diff')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('PSNR (dB)')
    plt.title('Sampler-agnostic PSNR (higher is better)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_samplers.pdf'), bbox_inches='tight')
    plt.close()

    print("Experiment 2 completed. Outputs saved in:", output_dir)
    return results


def experiment3(output_dir: str, quick: bool = True):
    ensure_dir(output_dir)
    set_seed(23)
    device = get_device()
    img_size = 32
    T = 30 if quick else 50
    sched = make_linear_schedule(T=T, device=device)
    ds_train, ds_val, dl_train, dl_val = create_dataloaders(512 if quick else 1024, 128 if quick else 256, img_size, batch_size=64)
    canonical_set = make_canonical_set(n=48, img_size=img_size)
    cond_vecs = torch.stack([c for _, c, _, _ in canonical_set]).to(device)
    x0_canon = torch.stack([x for x,_,_,_ in canonical_set]).to(device)

    t_embed = SinusoidalTimeEmbedding(dim=64)
    cond_mlp = GlobalCondMLP(cond_dim=8, t_dim=64, out_dim=128)

    teacher = BaselineUNet(1,16,128).to(device)
    train_baseline(teacher, dl_train, dl_val, sched, cond_mlp, t_embed, epochs=2 if quick else 3, lr=1e-3, device=device)

    configs = [
        {'name':'depth1_int8_learned', 'depth':1, 'quant':'int8', 'learned_gate':True, 'distill':True},
        {'name':'depth2_int8_learned', 'depth':2, 'quant':'int8', 'learned_gate':True, 'distill':True},
        {'name':'depth1_int4_learned', 'depth':1, 'quant':'int4', 'learned_gate':True, 'distill':True},
        {'name':'depth1_fp16_learned', 'depth':1, 'quant':'fp16', 'learned_gate':True, 'distill':True},
        {'name':'depth1_int8_manual', 'depth':1, 'quant':'int8', 'learned_gate':False, 'distill':True},
        {'name':'depth1_int8_scratch','depth':1, 'quant':'int8', 'learned_gate':True, 'distill':False},
    ]

    results = []

    for cfg in configs:
        print(f"Training ablation config: {cfg}")
        r2 = R2DiffUNet(1,16,128, delta_depth=cfg['depth'], quant_mode=cfg['quant']).to(device)
        if not cfg['learned_gate']:
            for g in r2.encoder_recur.gates:
                for p in g.parameters():
                    p.requires_grad = False
        if cfg['distill']:
            logs = train_r2diff(r2, teacher, dl_train, dl_val, sched, cond_mlp, t_embed, epochs=1 if quick else 2, lr=1e-3, device=device)
        else:
            opt = torch.optim.Adam(r2.parameters(), lr=1e-3)
            for _ in range(1 if quick else 2):
                for x0, cond, _, _ in dl_train:
                    x0 = x0.to(device); cond = cond.to(device)
                    B = x0.size(0)
                    t0 = torch.full((B,), T-1, device=device, dtype=torch.long)
                    x_t, eps_gt = q_sample(x0, t0, sched)
                    cond_emb = cond_mlp(cond, t_embed(t0))
                    eps1, fq, sc, zp = r2.forward_first(x_t, cond_emb)
                    loss = F.mse_loss(eps1, eps_gt)
                    opt.zero_grad(); loss.backward(); opt.step()
        imgs_r = sample_images_r2(r2, cond_mlp, t_embed, sched, cond_vecs, sampler='ddim', steps=10 if quick else 15, device=device)
        mse = F.mse_loss(imgs_r, x0_canon).item()
        psnr = -10.0 * math.log10(mse + 1e-12)
        flops = float('nan')
        if _HAS_FVCORE:
            try:
                h = torch.randn(1,16,32,32, device=device).half()
                cond_emb = torch.randn(1,128, device=device)
                fca = FlopCountAnalysis(r2.encoder_recur.delta_blocks[0], (h, cond_emb))
                flops = float(fca.total())
            except Exception:
                pass
        results.append({'config':cfg['name'], 'psnr':psnr, 'mse':mse, 'flops':flops})
        print(f"Ablation {cfg['name']} -> PSNR={psnr:.2f} dB, FLOPs_delta_block={flops}")

    labels = [r['config'] for r in results]
    values = [r['psnr'] for r in results]
    plot_bars(labels, values, 'Ablation PSNR (higher better)', 'PSNR (dB)', os.path.join(output_dir, 'accuracy_ablations.pdf'))

    print("Experiment 3 completed. Outputs saved in:", output_dir)
    return results
