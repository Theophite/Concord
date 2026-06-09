"""Minimal `train_cifar` providing the data loaders, evaluate(), and a
BaselineConvNet matching the concord architecture. The rest of the
concord training code imports from here for these utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BaselineConvNet(nn.Module):
    """Same architecture as ConcordConvNet but with standard nn.Conv2d/Linear.
    Used for AdamW/SGD baselines."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Stub: some training scripts check `isinstance(model, ConcordConvNet)`
# before calling sync_weights. The fused training path uses its own
# `FusedConvNet` (defined in train_cifar_fused.py) that ducks this check;
# this stub exists so the isinstance check doesn't crash on import.
class ConcordConvNet:
    pass


def get_loaders(batch_size=128, data_dir='./cifar_data', num_workers=4):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10(data_dir, train=True, download=True,
                             transform=tfm_train)
    test = datasets.CIFAR10(data_dir, train=False, download=True,
                            transform=tfm_test)
    persistent = num_workers > 0
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=persistent),
            DataLoader(test, batch_size=256, shuffle=False,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=persistent))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    if hasattr(model, 'sync_weights'):
        model.sync_weights()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total
