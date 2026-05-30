"""Minimal test: can packed-B layers be captured in a CUDA graph?

Start with the simplest possible case (single layer, no BN, no opt),
then add complexity until we find what breaks capture.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from prototype_packed_b import ConcordLinearPackedB, ConcordConv2dPackedB


def test_single_linear():
    print("=== Test 1: single ConcordLinearPackedB, AdamW ===")
    torch.manual_seed(0)
    layer = ConcordLinearPackedB(128, 64, device='cuda', lr=0.01)
    layer.set_optimizer_kind('adamw', weight_decay=0.01, eps=1.0)
    static_x = torch.randn(8, 128, device='cuda', dtype=torch.bfloat16,
                            requires_grad=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_x.grad = None
            y = layer(static_x)
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    static_x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        y = layer(static_x)
        loss = y.square().mean()
        loss.backward()
    print("  PASS: single linear captured")
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()


def test_two_linears():
    print("=== Test 2: two ConcordLinearPackedB stacked, AdamW ===")
    torch.manual_seed(0)
    layer1 = ConcordLinearPackedB(128, 64, device='cuda', lr=0.01)
    layer2 = ConcordLinearPackedB(64, 32, device='cuda', lr=0.01)
    for m in (layer1, layer2):
        m.set_optimizer_kind('adamw', weight_decay=0.01, eps=1.0)
    static_x = torch.randn(8, 128, device='cuda', dtype=torch.bfloat16,
                            requires_grad=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_x.grad = None
            y = layer2(layer1(static_x))
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    static_x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        y = layer2(layer1(static_x))
        loss = y.square().mean()
        loss.backward()
    print("  PASS: two linears captured")
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()


def test_conv_bn():
    print("=== Test 3: Conv2d + BN, AdamW ===")
    torch.manual_seed(0)
    conv = ConcordConv2dPackedB(3, 16, 3, padding=1, device='cuda', lr=0.01)
    bn = nn.BatchNorm2d(16).to('cuda')
    static_x = torch.randn(8, 3, 32, 32, device='cuda', dtype=torch.bfloat16,
                            requires_grad=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            static_x.grad = None
            for p in bn.parameters():
                if p.grad is not None:
                    p.grad = None
            y = bn(conv(static_x).float()).float()
            y.square().mean().backward()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    static_x.grad = None
    for p in bn.parameters():
        if p.grad is not None:
            p.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        y = bn(conv(static_x).float()).float()
        loss = y.square().mean()
        loss.backward()
    print("  PASS: Conv+BN captured")
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()


def test_with_sgd_opt():
    print("=== Test 4: Conv+BN + torch.optim.SGD on BN params ===")
    torch.manual_seed(0)
    conv = ConcordConv2dPackedB(3, 16, 3, padding=1, device='cuda', lr=0.01)
    bn = nn.BatchNorm2d(16).to('cuda')
    opt = torch.optim.SGD(bn.parameters(), lr=0.01, momentum=0.0)
    static_x = torch.randn(8, 3, 32, 32, device='cuda', dtype=torch.bfloat16,
                            requires_grad=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            static_x.grad = None
            y = bn(conv(static_x).float()).float()
            y.square().mean().backward()
            opt.step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    opt.zero_grad(set_to_none=True)
    static_x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        opt.zero_grad(set_to_none=False)
        y = bn(conv(static_x).float()).float()
        loss = y.square().mean()
        loss.backward()
        opt.step()
    print("  PASS: Conv+BN+SGD captured")
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)
    test_single_linear()
    test_two_linears()
    test_conv_bn()
    test_with_sgd_opt()
    print()
    print("[OK] all tests passed")
