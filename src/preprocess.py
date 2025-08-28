import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SyntheticImageDataset(Dataset):
    def __init__(self, n: int = 2048, img_size: int = 32, pattern: str = 'edges', n_classes: int = 10, class_conditional: bool = False):
        super().__init__()
        self.n = n
        self.img_size = img_size
        self.pattern = pattern
        self.n_classes = n_classes
        self.class_conditional = class_conditional
        self.data, self.labels = self._generate()

    def _generate(self):
        H = W = self.img_size
        data = []
        labels = []
        for _ in range(self.n):
            img = np.zeros((H, W, 3), dtype=np.float32)
            if self.pattern == 'edges':
                # draw random lines and rectangles
                for _ in range(5):
                    x1, y1 = np.random.randint(0, W), np.random.randint(0, H)
                    x2, y2 = np.random.randint(0, W), np.random.randint(0, H)
                    c = np.random.rand(3)
                    t = np.linspace(0, 1, 50)
                    xs = (x1 + (x2 - x1) * t).astype(int).clip(0, W - 1)
                    ys = (y1 + (y2 - y1) * t).astype(int).clip(0, H - 1)
                    img[ys, xs] = c
                # rectangle
                x1, y1 = np.random.randint(0, W//2), np.random.randint(0, H//2)
                x2, y2 = np.random.randint(W//2, W), np.random.randint(H//2, H)
                color = np.random.rand(3)
                img[y1:y2, x1:x2] = color
            elif self.pattern == 'textures':
                img = np.random.rand(H, W, 3).astype(np.float32)
                for _ in range(2):
                    img = (img + np.roll(img, 1, axis=0) + np.roll(img, -1, axis=0) + np.roll(img, 1, axis=1) + np.roll(img, -1, axis=1)) / 5.0
            else:  # 'blobs'
                img = np.zeros((H, W, 3), dtype=np.float32)
                for _ in range(10):
                    cx, cy = np.random.randint(0, W), np.random.randint(0, H)
                    r = np.random.randint(H//16, H//6)
                    Y, X = np.ogrid[:H, :W]
                    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
                    img[mask] = np.random.rand(3)
            lbl = np.random.randint(0, self.n_classes)
            data.append(img.transpose(2, 0, 1))
            labels.append(lbl)
        return np.stack(data), np.array(labels, dtype=np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])  # [3,H,W]
        y = int(self.labels[idx])
        return x, y


def make_noisy_labels(labels: torch.Tensor, noise_rate: float, n_classes: int) -> torch.Tensor:
    noisy = labels.clone()
    mask = torch.rand_like(noisy.float()) < noise_rate
    rand = torch.randint(0, n_classes, size=labels.shape, device=labels.device)
    noisy[mask] = rand[mask]
    return noisy
