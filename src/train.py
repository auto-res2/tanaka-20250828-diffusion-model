import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import (
    DiffusionSchedule,
)


# ------------------------------
# Model Components (Baseline UNet)
# ------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d_embed=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.film = nn.Sequential(
            nn.Linear(d_embed, 2*out_ch), nn.SiLU(), nn.Linear(2*out_ch, 2*out_ch)
        )

        
    def forward(self, x, cond_emb):
        x = self.conv1(x)
        x = self.norm1(x)
        gb = self.film(cond_emb)
        g, b = gb.chunk(2, dim=-1)
        x = x * (1 + g[..., None, None]) + b[..., None, None]
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        return x


class EncoderUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, d_embed=128):
        super().__init__()
        self.b1 = ConvBlock(in_ch, base_ch, d_embed)
        self.b2 = ConvBlock(base_ch, base_ch*2, d_embed)
        self.b3 = ConvBlock(base_ch*2, base_ch*2, d_embed)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, cond_emb):
        h1 = self.b1(x, cond_emb)
        p1 = self.pool(h1)
        h2 = self.b2(p1, cond_emb)
        p2 = self.pool(h2)
        h3 = self.b3(p2, cond_emb)
        return [h1, h2, h3]


class DecoderUNet(nn.Module):
    def __init__(self, out_ch=1, base_ch=16, d_embed=128):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch*2, 2, stride=2)
        self.bu1 = ConvBlock(base_ch*4, base_ch*2, d_embed)
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.bu2 = ConvBlock(base_ch*2, base_ch, d_embed)
        self.head = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x_in, feats: List[torch.Tensor], cond_emb):
        h1, h2, h3 = feats
        u1 = self.up1(h3)
        u1 = torch.cat([u1, h2], dim=1)
        u1 = self.bu1(u1, cond_emb)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, h1], dim=1)
        u2 = self.bu2(u2, cond_emb)
        eps = self.head(u2)
        return eps


class BaselineUNet(nn.Module):
    def __init__(self, img_ch=1, base_ch=16, d_embed=128):
        super().__init__()
        self.encoder = EncoderUNet(img_ch, base_ch, d_embed)
        self.decoder = DecoderUNet(img_ch, base_ch, d_embed)

    def forward(self, x_t, cond_emb):
        feats = self.encoder(x_t, cond_emb)
        eps = self.decoder(x_t, feats, cond_emb)
        return eps


# ------------------------------
# R2Diff Components
# ------------------------------

class DepthwiseConvGRU(nn.Module):
    def __init__(self, C, k=3):
        super().__init__()
        p = k // 2
        self.conv_z = nn.Conv2d(C*2, C, k, padding=p, groups=C)
        self.conv_r = nn.Conv2d(C*2, C, k, padding=p, groups=C)
        self.conv_h = nn.Conv2d(C*2, C, k, padding=p, groups=C)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        rh = torch.cat([x, r*h], dim=1)
        h_tilde = torch.tanh(self.conv_h(rh))
        return (1 - z) * h + z * h_tilde


class FiLM(nn.Module):
    def __init__(self, C, d_embed=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_embed, 2*C), nn.SiLU(), nn.Linear(2*C, 2*C)
        )

    def forward(self, x, cond):
        gb = self.proj(cond)
        g, b = gb.chunk(2, dim=-1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class DeltaBlock(nn.Module):
    def __init__(self, C, d_embed=128, depth=1):
        super().__init__()
        self.grus = nn.ModuleList([DepthwiseConvGRU(C) for _ in range(depth)])
        self.film = FiLM(C, d_embed)
        self.out = nn.Conv2d(C, C, 1)

    def forward(self, h_prev, cond):
        h = h_prev
        for gru in self.grus:
            h = gru(h_prev, h)
        h = self.film(h, cond)
        delta = self.out(h)
        return delta


class TimeGating(nn.Module):
    def __init__(self, d_embed=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, d_embed//2), nn.SiLU(),
            nn.Linear(d_embed//2, 1), nn.Sigmoid()
        )

    def forward(self, cond_emb):
        return self.net(cond_emb).view(-1, 1, 1, 1)


class PerChannelUint8:
    @staticmethod
    def quantize(x: torch.Tensor, pct=0.999):
        x = x.float()
        B,C,H,W = x.shape
        xf = x.permute(0,2,3,1).reshape(-1,C)
        lo = torch.quantile(xf, 1-pct, dim=0)
        hi = torch.quantile(xf, pct, dim=0)
        scale = (hi - lo).clamp(min=1e-6) / 255.0
        zp = (-lo / scale).round().clamp(0,255)
        q = (x.permute(0,2,3,1) / scale - zp).round().clamp(0,255).to(torch.uint8)
        q = q.permute(0,3,1,2).contiguous()
        return q, scale, zp

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
        return (q.float() + zp[None,:,None,None]) * scale[None,:,None,None]


class PerChannelUint4:
    @staticmethod
    def quantize(x: torch.Tensor, pct=0.999):
        x = x.float()
        B,C,H,W = x.shape
        xf = x.permute(0,2,3,1).reshape(-1,C)
        lo = torch.quantile(xf, 1-pct, dim=0)
        hi = torch.quantile(xf, pct, dim=0)
        scale = (hi - lo).clamp(min=1e-6) / 15.0
        zp = (-lo / scale).round().clamp(0,15)
        q = (x.permute(0,2,3,1) / scale - zp).round().clamp(0,15).to(torch.uint8)
        q = q.permute(0,3,1,2).contiguous()
        return q, scale, zp

    @staticmethod
    def dequantize(q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor):
        return (q.float() + zp[None,:,None,None]) * scale[None,:,None,None]


class R2DiffEncoder(nn.Module):
    def __init__(self, encoder: EncoderUNet, ch_layout: List[int], d_embed=128, depth=1, quant_mode='int8'):
        super().__init__()
        self.ote = encoder
        self.delta_blocks = nn.ModuleList([DeltaBlock(C, d_embed, depth) for C in ch_layout])
        self.gates = nn.ModuleList([TimeGating(d_embed) for _ in ch_layout])
        self.quant_mode = quant_mode

    @torch.no_grad()
    def run_ote(self, x, cond_emb):
        feats = self.ote(x, cond_emb)
        return feats

    def _quant(self, h):
        if self.quant_mode == 'int8':
            return PerChannelUint8.quantize(h)
        elif self.quant_mode == 'int4':
            return PerChannelUint4.quantize(h)
        else:
            return h.float(), torch.tensor([], device=h.device), torch.tensor([], device=h.device)

    def _dequant(self, q, sc, zp):
        if self.quant_mode == 'int8':
            return PerChannelUint8.dequantize(q, sc, zp).float()
        elif self.quant_mode == 'int4':
            return PerChannelUint4.dequantize(q, sc, zp).float()
        else:
            return q.float()

    def quantize_feats(self, feats: List[torch.Tensor]):
        feats_q, scales, zps = [], [], []
        for f in feats:
            q, sc, zp = self._quant(f)
            feats_q.append(q); scales.append(sc); zps.append(zp)
        return feats_q, scales, zps

    def recurrent_update(self, feats_q, scales, zps, cond_emb):
        new_q, new_sc, new_zp = [], [], []
        for (q, sc, zp, db, gate) in zip(feats_q, scales, zps, self.delta_blocks, self.gates):
            h = self._dequant(q, sc.to(q.device), zp.to(q.device))
            delta = db(h, cond_emb) * gate(cond_emb)
            h_new = (h + delta).float()
            q2, sc2, zp2 = self._quant(h_new)
            new_q.append(q2); new_sc.append(sc2); new_zp.append(zp2)
        return new_q, new_sc, new_zp


class R2DiffUNet(nn.Module):
    def __init__(self, img_ch=1, base_ch=16, d_embed=128, delta_depth=1, quant_mode='int8'):
        super().__init__()
        self.encoder_core = EncoderUNet(img_ch, base_ch, d_embed)
        self.decoder = DecoderUNet(img_ch, base_ch, d_embed)
        ch_layout = [base_ch, base_ch*2, base_ch*2]
        self.encoder_recur = R2DiffEncoder(self.encoder_core, ch_layout, d_embed, delta_depth, quant_mode)

    def forward_first(self, x_t, cond_emb):
        feats = self.encoder_recur.run_ote(x_t, cond_emb)
        feats_q, scales, zps = self.encoder_recur.quantize_feats(feats)
        eps = self.decoder(x_t, feats, cond_emb)
        return eps, feats_q, scales, zps

    def forward_recurrent(self, x_t, cond_emb, feats_q, scales, zps):
        feats_q, scales, zps = self.encoder_recur.recurrent_update(feats_q, scales, zps, cond_emb)
        feats = [self.encoder_recur._dequant(q, sc.to(x_t.device), zp.to(x_t.device)) for q, sc, zp in zip(feats_q, scales, zps)]
        eps = self.decoder(x_t, feats, cond_emb)
        return eps, feats_q, scales, zps


# ------------------------------
# Token (DiT-like) backbone and R2 variant (toy)
# ------------------------------

class TokenBackbone(nn.Module):
    def __init__(self, img_size=32, patch=4, dim=64, d_embed=128):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.dim = dim
        self.proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.enc1 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.enc2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch*patch))
        self.cond = nn.Sequential(nn.Linear(d_embed, dim), nn.SiLU())

    def forward(self, x_t, cond_emb):
        B, _, H, W = x_t.shape
        tokens = self.proj(x_t).permute(0,2,3,1).reshape(B, -1, self.dim)
        c = self.cond(cond_emb).unsqueeze(1)
        h1 = self.enc1(tokens + c)
        h2 = self.enc2(h1 + c)
        out = self.head(h2).reshape(B, H//self.patch, W//self.patch, self.patch, self.patch)
        out = out.permute(0,3,4,1,2).reshape(B,1,H,W)
        return out


class TokenDeltaBlock(nn.Module):
    def __init__(self, C=64, d_embed=128, depth=1):
        super().__init__()
        self.grus = nn.ModuleList([nn.GRU(input_size=C, hidden_size=C, num_layers=1, batch_first=True) for _ in range(depth)])
        self.film = nn.Sequential(nn.Linear(d_embed, 2*C), nn.SiLU(), nn.Linear(2*C, 2*C))
        self.out = nn.Linear(C, C)

    def forward(self, tokens, cond_emb):
        h = tokens
        for gru in self.grus:
            h,_ = gru(h)
        gb = self.film(cond_emb)
        g,b = gb.chunk(2,-1)
        h = h * (1 + g[:,None,:]) + b[:,None,:]
        return self.out(h)


class R2DiffToken(nn.Module):
    def __init__(self, img_size=32, patch=4, dim=64, d_embed=128, delta_depth=1, quant_mode='int8'):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.dim = dim
        self.ote = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.enc1 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.enc2 = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU())
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch*patch))
        self.cond = nn.Sequential(nn.Linear(d_embed, dim), nn.SiLU())
        self.delta = TokenDeltaBlock(dim, d_embed, delta_depth)
        self.gate = TimeGating(d_embed)
        self.quant_mode = quant_mode

    def _quant_tok(self, tok):
        x = tok.float()
        lo = torch.quantile(x, 0.001, dim=1)
        hi = torch.quantile(x, 0.999, dim=1)
        if self.quant_mode == 'int8':
            scale = (hi - lo).clamp(min=1e-6) / 255.0
            zp = (-lo / scale).round().clamp(0,255)
            q = (x / scale[:,None,:] - zp[:,None,:]).round().clamp(0,255).to(torch.uint8)
            return q, scale, zp
        elif self.quant_mode == 'int4':
            scale = (hi - lo).clamp(min=1e-6) / 15.0
            zp = (-lo / scale).round().clamp(0,15)
            q = (x / scale[:,None,:] - zp[:,None,:]).round().clamp(0,15).to(torch.uint8)
            return q, scale, zp
        else:
            return x.float(), torch.tensor([], device=x.device), torch.tensor([], device=x.device)

    def _dequant_tok(self, q, sc, zp):
        if self.quant_mode in ['int8','int4']:
            return (q.float() + zp[:,None,:]) * sc[:,None,:]
        else:
            return q.float()

    def forward_first(self, x_t, cond_emb):
        B,_,H,W = x_t.shape
        tokens = self.ote(x_t).permute(0,2,3,1).reshape(B, -1, self.dim)
        c = self.cond(cond_emb).unsqueeze(1)
        t1 = self.enc1(tokens + c)
        t2 = self.enc2(t1 + c)
        q, sc, zp = self._quant_tok(t2)
        out = self.head(t2).reshape(B, H//self.patch, W//self.patch, self.patch, self.patch)
        eps = out.permute(0,3,4,1,2).reshape(B,1,H,W)
        return eps, q, sc, zp

    def forward_recurrent(self, x_t, cond_emb, q, sc, zp):
        B,_,H,W = x_t.shape
        t2 = self._dequant_tok(q, sc.to(x_t.device), zp.to(x_t.device))
        # Gate is per-batch scalar; reshape to broadcast over (B, N, C)
        g = self.gate(cond_emb).squeeze(-1)  # (B, 1, 1)
        delta = self.delta(t2, cond_emb) * g
        t2_new = (t2 + delta)
        q2, sc2, zp2 = self._quant_tok(t2_new)
        out = self.head(t2_new).reshape(B, H//self.patch, W//self.patch, self.patch, self.patch)
        eps = out.permute(0,3,4,1,2).reshape(B,1,H,W)
        return eps, q2, sc2, zp2


# ------------------------------
# Training helpers
# ------------------------------

def validate_epsilon_mse(model, dl_val, sched: DiffusionSchedule, cond_mlp, t_embed, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x0, cond, _, _ in dl_val:
            x0 = x0.to(device); cond = cond.to(device)
            B = x0.size(0)
            t = torch.randint(low=0, high=sched.betas.shape[0], size=(B,), device=device)
            x_t, eps = q_sample(x0, t, sched)
            cond_emb = cond_mlp(cond, t_embed(t))
            pred = model(x_t, cond_emb)
            losses.append(F.mse_loss(pred, eps).item())
    model.train()
    return float(np.mean(losses))


def train_baseline(model: nn.Module, dl_train, dl_val, sched: DiffusionSchedule,
                   cond_mlp, t_embed,
                   epochs=2, lr=1e-3, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    cond_mlp.to(device)
    t_embed.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [] , []
    model.train()
    for ep in range(epochs):
        for x0, cond, _, _ in dl_train:
            x0 = x0.to(device)
            cond = cond.to(device)
            B = x0.size(0)
            t = torch.randint(low=0, high=sched.betas.shape[0], size=(B,), device=device)
            x_t, eps = q_sample(x0, t, sched)
            t_emb = t_embed(t)
            cond_emb = cond_mlp(cond, t_emb)
            pred = model(x_t, cond_emb)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        vl = validate_epsilon_mse(model, dl_val, sched, cond_mlp, t_embed, device)
        val_losses.append(vl)
        print(f"[Baseline] Epoch {ep+1}/{epochs} val_eps_mse={vl:.4f}")
    return train_losses, val_losses


@dataclass
class R2TrainLogs:
    train_losses: List[float]
    val_losses: List[float]
    gate_means: List[float]


def q_sample(x0: torch.Tensor, t: torch.Tensor, sched: DiffusionSchedule, noise: Optional[torch.Tensor] = None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ab = sched.sqrt_alphas_bar[t].view(-1, 1, 1, 1)
    sqrt_omab = sched.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1)
    return sqrt_ab * x0 + sqrt_omab * noise, noise


def validate_epsilon_mse_r2(model_r2, teacher, dl_val, sched, cond_mlp, t_embed, device) -> float:
    model_r2.eval(); teacher.eval()
    losses = []
    with torch.no_grad():
        for x0, cond, _, _ in dl_val:
            x0 = x0.to(device); cond = cond.to(device)
            B = x0.size(0); T = sched.betas.shape[0]
            t0 = torch.full((B,), T-1, device=device, dtype=torch.long)
            x_t, eps_gt = q_sample(x0, t0, sched)
            cond_emb = cond_mlp(cond, t_embed(t0))
            eps_pred, feats_q, scales, zps = model_r2.forward_first(x_t, cond_emb)
            l = F.mse_loss(eps_pred, eps_gt).item()
            losses.append(l)
    model_r2.train(); teacher.train()
    return float(np.mean(losses))


def train_r2diff(model_r2: R2DiffUNet, teacher: BaselineUNet, dl_train, dl_val, sched: DiffusionSchedule,
                 cond_mlp, t_embed,
                 epochs=2, lr=1e-3, device=None, K_range=(2,3), distill_w=0.25, gate_reg=1e-3) -> R2TrainLogs:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model_r2.to(device)
    cond_mlp.to(device)
    t_embed.to(device)
    teacher.to(device).eval()
    opt = torch.optim.Adam(model_r2.parameters(), lr=lr)
    train_losses, val_losses, gate_means = [], [], []
    for ep in range(epochs):
        for x0, cond, _, _ in dl_train:
            x0 = x0.to(device); cond = cond.to(device)
            B = x0.size(0)
            T = sched.betas.shape[0]
            K = random.randint(K_range[0], K_range[1])
            s = random.randint(0, max(0, T-K))
            t_seq = list(range(T-1-s, T-1-s-K, -1))
            t0 = torch.full((B,), t_seq[0], device=device, dtype=torch.long)
            x_t, eps_gt0 = q_sample(x0, t0, sched)
            cond_emb = cond_mlp(cond, t_embed(t0))
            eps_pred, feats_q, scales, zps = model_r2.forward_first(x_t, cond_emb)
            with torch.no_grad():
                eps_teacher = teacher(x_t, cond_emb)
            loss = F.mse_loss(eps_pred, eps_gt0) + distill_w * F.mse_loss(eps_pred, eps_teacher)
            mean_gate = 0.0
            for t_idx in t_seq[1:]:
                t_cur = torch.full((B,), t_idx, device=device, dtype=torch.long)
                cond_emb = cond_mlp(cond, t_embed(t_cur))
                eps_pred, feats_q, scales, zps = model_r2.forward_recurrent(x_t, cond_emb, feats_q, scales, zps)
                with torch.no_grad():
                    eps_teacher = teacher(x_t, cond_emb)
                    _, eps_gt = q_sample(x0, t_cur, sched)
                gm = 0.0
                for g in model_r2.encoder_recur.gates:
                    gm = gm + g(cond_emb).mean()
                gm = gm / len(model_r2.encoder_recur.gates)
                mean_gate = mean_gate + gm.item()
                loss = loss + F.mse_loss(eps_pred, eps_gt) + distill_w * F.mse_loss(eps_pred, eps_teacher) + gate_reg * gm
            mean_gate = mean_gate / max(1, (len(t_seq)-1))
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
            gate_means.append(mean_gate)
        vl = validate_epsilon_mse_r2(model_r2, teacher, dl_val, sched, cond_mlp, t_embed, device)
        val_losses.append(vl)
        print(f"[R2Diff] Epoch {ep+1}/{epochs} val_eps_mse={vl:.4f} avg_gate={np.mean(gate_means[-len(dl_train):]):.3f}")
    return R2TrainLogs(train_losses, val_losses, gate_means)
