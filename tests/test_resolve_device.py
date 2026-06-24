"""Tests for cross-platform compute-device resolution.

resolve_device() must honor an explicit TORCH_DEVICE override and otherwise
auto-detect, falling back to 'cpu' on machines with neither MPS nor CUDA
(e.g. a plain Windows or Linux box) instead of hardcoding 'mps'.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _pipeline_core import resolve_device


def test_explicit_override_is_honored(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE", "cuda")
    assert resolve_device() == "cuda"


def test_explicit_override_cpu(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_autodetect_falls_back_to_cpu_when_no_accelerator(monkeypatch):
    """No override, no MPS, no CUDA -> 'cpu', not 'mps'."""
    monkeypatch.delenv("TORCH_DEVICE", raising=False)

    fake_torch = type("T", (), {})()
    fake_torch.cuda = type("C", (), {"is_available": staticmethod(lambda: False)})()
    fake_torch.backends = type(
        "B", (), {"mps": type("M", (), {"is_available": staticmethod(lambda: False)})()}
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device() == "cpu"


def test_autodetect_prefers_cuda_when_available(monkeypatch):
    monkeypatch.delenv("TORCH_DEVICE", raising=False)

    fake_torch = type("T", (), {})()
    fake_torch.cuda = type("C", (), {"is_available": staticmethod(lambda: True)})()
    fake_torch.backends = type(
        "B", (), {"mps": type("M", (), {"is_available": staticmethod(lambda: False)})()}
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device() == "cuda"


def test_autodetect_uses_mps_when_only_mps(monkeypatch):
    monkeypatch.delenv("TORCH_DEVICE", raising=False)

    fake_torch = type("T", (), {})()
    fake_torch.cuda = type("C", (), {"is_available": staticmethod(lambda: False)})()
    fake_torch.backends = type(
        "B", (), {"mps": type("M", (), {"is_available": staticmethod(lambda: True)})()}
    )()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device() == "mps"


def test_autodetect_without_torch_installed_returns_cpu(monkeypatch):
    """If torch import fails entirely, degrade to 'cpu' rather than raise."""
    monkeypatch.delenv("TORCH_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`
    assert resolve_device() == "cpu"