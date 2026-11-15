"""Regression test: XOR loss accepts CPU batches for a model on the GPU."""

from __future__ import annotations

import pytest
import torch

from core.xor import XorMLP, xor_loss


def test_xor_loss_moves_batch_to_model_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")
    model = XorMLP(in_dim=2, hidden_size=8).to("cuda")
    batch = {
        "x": torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "y": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    }
    loss, metrics = xor_loss(model, batch)
    assert loss.device.type == "cuda"
    assert 0.0 <= metrics["acc"] <= 1.0
