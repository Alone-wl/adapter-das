"""Cross-framework bootstrap guard ordering and isolation.

Run: python3 -m pytest -q tests/adapter/test_common_bridge.py
"""

from types import SimpleNamespace

from hcu_adapter.bridges import common
import hcu_adapter


def test_ar_norm_install_defers_vendor_import_and_preserves_stock_fallback():
    import sys
    from types import ModuleType
    from unittest.mock import Mock, patch
    from hcu_adapter import ar_norm_aiter

    modules = {}
    for name in (
        "sglang", "sglang.srt", "sglang.srt.layers",
        "sglang.srt.layers.flashinfer_comm_fusion",
        "sglang.srt.layers.layer_boundary", "sglang.srt.layers.layer_boundary.exit",
        "sglang.srt.layers.layer_boundary.fusions",
        "sglang.srt.layers.layer_boundary.fusions.allreduce",
        "sglang.srt.layers.layer_boundary.residual",
        "sglang.srt.layers.layer_boundary.residual.add_norm",
        "sglang.srt.distributed",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    fusion = modules["sglang.srt.layers.flashinfer_comm_fusion"]
    stock = Mock(return_value=object())
    stock._hcu_ar_norm_aiter = False
    fusion.flashinfer_allreduce_residual_rmsnorm = stock
    modules["sglang.srt.distributed"].get_tp_group = lambda: SimpleNamespace(ca_comm=None)
    with patch.dict(sys.modules, {
        **modules, "torch": SimpleNamespace(version=SimpleNamespace(hip="6.4")),
        "aiter": None,
    }), patch.object(ar_norm_aiter.hcu_env, "resolve", return_value=("aiter", "auto")), patch.object(
        ar_norm_aiter, "register"
    ):
        ar_norm_aiter.install()
        assert fusion.flashinfer_allreduce_residual_rmsnorm is not stock
        assert fusion.flashinfer_allreduce_residual_rmsnorm(1, 2, 3) is stock.return_value
    stock.assert_called_once()


def test_common_guards_run_in_pre_import_order(monkeypatch):
    events = []
    monkeypatch.setattr(
        hcu_adapter,
        "hcu_env",
        SimpleNamespace(log_summary=lambda: events.append("environment")),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "nccl_topo_guard",
        SimpleNamespace(install=lambda: events.append("nccl")),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "tvmffi_warmup",
        SimpleNamespace(install=lambda: events.append("tvmffi")),
        raising=False,
    )

    common.install()

    assert events == ["environment", "nccl", "tvmffi"]


def test_one_common_guard_failure_does_not_block_later_guards(monkeypatch):
    events = []

    def fail_environment():
        events.append("environment")
        raise RuntimeError("summary failed")

    monkeypatch.setattr(
        hcu_adapter,
        "hcu_env",
        SimpleNamespace(log_summary=fail_environment),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "nccl_topo_guard",
        SimpleNamespace(install=lambda: events.append("nccl")),
        raising=False,
    )
    monkeypatch.setattr(
        hcu_adapter,
        "tvmffi_warmup",
        SimpleNamespace(install=lambda: events.append("tvmffi")),
        raising=False,
    )

    common.install()

    assert events == ["environment", "nccl", "tvmffi"]
