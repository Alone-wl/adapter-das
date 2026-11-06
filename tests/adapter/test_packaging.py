import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_weakref_import_selects_native_symbol_without_cuda_allocation():
    import importlib.util
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    class Tensor:
        device = "cuda:3"

        def numel(self):
            return 8

    native = Mock(return_value=object())
    zeros = Mock(return_value=Tensor())
    torch = SimpleNamespace(
        Tensor=Tensor, zeros=zeros,
        ops=SimpleNamespace(sgl_kernel=SimpleNamespace(weak_ref_tensor=native)),
    )
    with patch.dict(sys.modules, {
        "torch": torch, "sgl_kernel.common_ops": SimpleNamespace(),
    }):
        spec = importlib.util.spec_from_file_location(
            "weakref_import_contract", REPO_ROOT / "sgl_kernel/weak_ref.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        zeros.assert_not_called()
        native.assert_not_called()
        empty = Tensor()
        empty.numel = lambda: 0
        assert module.weak_ref_tensor(empty) is empty
        zeros.assert_not_called()
        tensor = Tensor()
        assert module.weak_ref_tensor(tensor) is native.return_value
        assert native.call_args_list[0].args == (zeros.return_value,)
        assert native.call_args_list[1].args == (tensor,)
        zeros.assert_called_once_with(1, device="cuda:3")
        assert module.weak_ref_tensor(tensor) is native.return_value
        assert native.call_count == 3
        zeros.assert_called_once()
        assert module.weak_ref_tensor(None) is None


def test_weakref_failed_native_probe_keeps_view_fallback_on_tensor_device():
    import importlib.util
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    class Tensor:
        device = "cuda:3"

        def numel(self):
            return 8

        def view_as(self, other):
            assert other is self
            return self

    native = Mock(side_effect=ValueError("invalid device ABI"))
    alternate = Mock(side_effect=ValueError("invalid device ABI"))
    zeros = Mock(return_value=Tensor())
    torch = SimpleNamespace(
        Tensor=Tensor, zeros=zeros,
        ops=SimpleNamespace(sgl_kernel=SimpleNamespace(weak_ref_tensor=native)),
    )
    with patch.dict(sys.modules, {
        "torch": torch, "sgl_kernel.common_ops": SimpleNamespace(weak_ref_tensor=alternate),
    }):
        spec = importlib.util.spec_from_file_location(
            "weakref_fallback_contract", REPO_ROOT / "sgl_kernel/weak_ref.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        zeros.assert_not_called()
        tensor = Tensor()
        assert module.weak_ref_tensor(tensor) is tensor
        assert module.weak_ref_tensor(tensor) is tensor
        zeros.assert_called_once_with(1, device="cuda:3")
        native.assert_called_once()
        alternate.assert_called_once()


def test_wheel_contains_bootstrap_and_nested_adapter_packages(tmp_path):
    source = tmp_path / "source"
    wheel_dir = tmp_path / "wheel"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist", "*.egg-info"
        ),
    )
    wheel_dir.mkdir()
    env = os.environ.copy()
    env["HCU_ADAPTER_HOST"] = "none"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

    assert "hcu_adapter/extensions.py" in members

    expected = {"sitecustomize.py"}
    for package in ("hcu_adapter", "sgl_kernel"):
        expected.update(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / package).rglob("*.py")
        )

    assert not (expected - members), f"wheel is missing: {sorted(expected - members)}"


def test_vllm_official_entry_points_share_one_atomic_allowlist_name():
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    entry_points = project["entry-points"]
    assert entry_points["vllm.platform_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_platform_plugin"
    }
    assert entry_points["vllm.general_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
    }
