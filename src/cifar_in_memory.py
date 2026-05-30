"""In-memory CIFAR-10 loader with GPU-side augmentation -- no DataLoader
worker processes (the source of the persistent_workers hangs we kept hitting
on this Windows box).

Pre-loads the entire train and test sets as GPU uint8 tensors (~150 MB train
+ 30 MB test, negligible on a 24 GB card). Iterates batches via index
slicing; applies RandomCrop(padding=4) + RandomHorizontalFlip in vectorised
GPU ops; normalizes with the standard CIFAR-10 mean/std.

Drop-in replacement for the (train, test) DataLoader pair the rest of the
code expects: each loader is an iterable that yields (x, y) bf16 / int64
batches on the configured device.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@torch.no_grad()
def _random_crop_pad4(x):
    """x: (B, 3, 32, 32) fp32. Returns randomly cropped (B, 3, 32, 32) after
    zero-padding to 40x40. Per-sample independent offsets, vectorised."""
    B, C, _, _ = x.shape
    padded = F.pad(x, (4, 4, 4, 4))  # (B, 3, 40, 40)
    off_h = torch.randint(0, 9, (B,), device=x.device)
    off_w = torch.randint(0, 9, (B,), device=x.device)
    # Vectorised gather: build row/col indices into the padded tensor.
    row_arange = torch.arange(32, device=x.device).view(1, 1, 32, 1)
    col_arange = torch.arange(32, device=x.device).view(1, 1, 1, 32)
    rows = off_h.view(B, 1, 1, 1) + row_arange   # (B, 1, 32, 1)
    cols = off_w.view(B, 1, 1, 1) + col_arange   # (B, 1, 1, 32)
    # broadcast over channel dim
    rows = rows.expand(B, C, 32, 32)
    cols = cols.expand(B, C, 32, 32)
    batch_idx = (torch.arange(B, device=x.device)
                  .view(B, 1, 1, 1).expand(B, C, 32, 32))
    chan_idx = (torch.arange(C, device=x.device)
                  .view(1, C, 1, 1).expand(B, C, 32, 32))
    return padded[batch_idx, chan_idx, rows, cols]


@torch.no_grad()
def _random_hflip(x):
    """Per-sample independent horizontal flip with p=0.5."""
    B = x.shape[0]
    mask = (torch.rand(B, device=x.device) < 0.5).view(B, 1, 1, 1)
    flipped = torch.flip(x, dims=[-1])
    return torch.where(mask, flipped, x)


class CIFAR10InMemoryLoader:
    """Iterable that yields (x, y) batches on `device`. x is fp32 normalized
    (mean / std), shape (B, 3, 32, 32). y is int64 (B,). The augmentation
    flag controls whether RandomCrop + HFlip fire (on for train, off for
    test).
    """

    def __init__(self, x_uint8, y_int64, batch_size, device, augment,
                 shuffle):
        self.x = x_uint8.to(device)        # (n, 3, 32, 32) uint8
        self.y = y_int64.to(device)         # (n,) int64
        self.batch_size = int(batch_size)
        self.device = device
        self.augment = bool(augment)
        self.shuffle = bool(shuffle)
        m = torch.tensor(CIFAR10_MEAN, device=device, dtype=torch.float32)
        s = torch.tensor(CIFAR10_STD, device=device, dtype=torch.float32)
        self._mean = m.view(1, 3, 1, 1)
        self._std = s.view(1, 3, 1, 1)
        self._n = self.x.shape[0]

    def __len__(self):
        return (self._n + self.batch_size - 1) // self.batch_size

    @torch.no_grad()
    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self._n, device=self.device)
        else:
            idx = torch.arange(self._n, device=self.device)
        for i in range(0, self._n, self.batch_size):
            sl = idx[i:i + self.batch_size]
            x = self.x[sl].float() / 255.0      # (B, 3, 32, 32) fp32
            if self.augment:
                x = _random_crop_pad4(x)
                x = _random_hflip(x)
            x = (x - self._mean) / self._std
            yield x, self.y[sl]


def get_loaders_in_memory(batch_size, device, data_dir='./cifar_data'):
    """Build the (train, test) loader pair, pre-loading both splits to
    `device`. No worker processes -- avoids the Windows shared-file-mapping
    issues that have hung this experiment several times.
    """
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=None)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True,
                                transform=None)

    def to_tensors(ds):
        import numpy as np
        x_np = np.asarray(ds.data, dtype=np.uint8)               # (n,32,32,3)
        # Permute NHWC -> NCHW, contiguous so the GPU tensor is well-formed.
        x = (torch.from_numpy(x_np)
             .permute(0, 3, 1, 2).contiguous())                   # (n,3,32,32)
        y = torch.tensor(ds.targets, dtype=torch.int64)
        return x, y

    train_x, train_y = to_tensors(train_ds)
    test_x, test_y = to_tensors(test_ds)
    train = CIFAR10InMemoryLoader(train_x, train_y, batch_size, device,
                                    augment=True, shuffle=True)
    test = CIFAR10InMemoryLoader(test_x, test_y, 256, device,
                                   augment=False, shuffle=False)
    return train, test
