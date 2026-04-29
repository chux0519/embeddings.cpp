import importlib.util
from pathlib import Path

import setuptools


ROOT = Path(__file__).resolve().parents[1]


def load_setup_module(monkeypatch):
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: kwargs)
    spec = importlib.util.spec_from_file_location("embeddings_cpp_setup_under_test", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_riscv64_wheel_build_disables_unsupported_zvfh(monkeypatch):
    monkeypatch.setenv("CIBW_ARCHS", "riscv64")

    module = load_setup_module(monkeypatch)

    assert module._platform_cmake_args() == ["-DGGML_RV_ZVFH=OFF"]


def test_non_riscv64_wheel_build_keeps_default_cmake_args(monkeypatch):
    monkeypatch.delenv("CIBW_ARCHS", raising=False)
    monkeypatch.delenv("CIBW_BUILD", raising=False)
    monkeypatch.delenv("AUDITWHEEL_ARCH", raising=False)
    module = load_setup_module(monkeypatch)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")

    assert module._platform_cmake_args() == []
