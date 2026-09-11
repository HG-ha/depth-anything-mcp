import sys

from depth_anything_mcp.runtime import parse_device_spec, resolve_device_spec


def test_parse_device_spec() -> None:
    assert parse_device_spec("auto").kind == "auto"
    assert parse_device_spec("").kind == "auto"
    assert parse_device_spec("cpu").kind == "cpu"
    assert parse_device_spec("dml").kind == "dml"
    assert parse_device_spec("directml").kind == "dml"
    assert parse_device_spec("gpu").label == "cuda:0"
    assert parse_device_spec("cuda").label == "cuda:0"
    assert parse_device_spec("cuda:1").label == "cuda:1"
    assert parse_device_spec("0").label == "cuda:0"


def test_parse_device_spec_rejects_unknown() -> None:
    try:
        parse_device_spec("tpu")
    except ValueError as exc:
        assert "device must be" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_auto_windows_uses_directml(monkeypatch) -> None:
    monkeypatch.setattr("depth_anything_mcp.runtime.is_windows", lambda: True)
    monkeypatch.setattr("depth_anything_mcp.runtime.has_nvidia", lambda: False)
    assert resolve_device_spec("auto").kind == "dml"


def test_auto_linux_nvidia_uses_cuda(monkeypatch) -> None:
    monkeypatch.setattr("depth_anything_mcp.runtime.is_windows", lambda: False)
    monkeypatch.setattr("depth_anything_mcp.runtime.has_nvidia", lambda: True)
    assert resolve_device_spec("auto").label == "cuda:0"


def test_auto_linux_no_gpu_uses_cpu(monkeypatch) -> None:
    monkeypatch.setattr("depth_anything_mcp.runtime.is_windows", lambda: False)
    monkeypatch.setattr("depth_anything_mcp.runtime.has_nvidia", lambda: False)
    assert resolve_device_spec("auto").kind == "cpu"


def test_explicit_cpu_not_overridden(monkeypatch) -> None:
    monkeypatch.setattr("depth_anything_mcp.runtime.is_windows", lambda: True)
    monkeypatch.setattr("depth_anything_mcp.runtime.has_nvidia", lambda: True)
    assert resolve_device_spec("cpu").kind == "cpu"


def test_platform_helper_matches_sys() -> None:
    from depth_anything_mcp.runtime import is_windows

    assert is_windows() is (sys.platform == "win32")


def test_uv_uninstall_drops_yes_flag(monkeypatch) -> None:
    from depth_anything_mcp import runtime

    monkeypatch.setattr(runtime.shutil, "which", lambda name: r"C:\uv.exe" if name == "uv" else None)
    command = runtime._uv_pip_command(["uninstall", "-y", "onnxruntime", "onnxruntime-directml"])
    assert command is not None
    assert command[:3] == [r"C:\uv.exe", "pip", "uninstall"]
    assert "-y" not in command
    assert "onnxruntime" in command
    assert "onnxruntime-directml" in command


def test_auto_installs_directml_without_probing_ort_first(monkeypatch) -> None:
    from depth_anything_mcp import runtime

    runtime._ENSURED.clear()
    installed: list[str] = []

    def fake_install(spec: str, *, allow_imported: bool = False) -> None:
        installed.append(spec)

    def fake_has_ep(name: str) -> bool:
        raise AssertionError("must not import/probe ORT before the GPU wheel is installed")

    monkeypatch.setattr(runtime, "is_windows", lambda: True)
    monkeypatch.setattr(runtime, "has_nvidia", lambda: True)
    monkeypatch.setattr(runtime, "_installed_ort_packages", lambda: {"onnxruntime"})
    monkeypatch.setattr(runtime, "_install_wheel", fake_install)
    monkeypatch.setattr(runtime, "_has_ep", fake_has_ep)
    monkeypatch.setattr(
        runtime,
        "_ort_providers_safe",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"] if installed else ["CPUExecutionProvider"],
    )

    def after_install_has_ep(name: str) -> bool:
        return bool(installed) and name == "DmlExecutionProvider"

    # After install, ensure() checks providers; swap the probe at that point.
    def install_and_enable(spec: str, *, allow_imported: bool = False) -> None:
        installed.append(spec)
        monkeypatch.setattr(runtime, "_has_ep", after_install_has_ep)

    monkeypatch.setattr(runtime, "_install_wheel", install_and_enable)
    report = runtime.ensure_onnxruntime("auto")
    assert installed
    assert report["spec"].kind == "dml"


def test_ensure_onnxruntime_cpu_skips_install(monkeypatch) -> None:
    from depth_anything_mcp import runtime

    runtime._ENSURED.clear()
    monkeypatch.setattr(runtime, "_install_wheel", lambda spec: (_ for _ in ()).throw(RuntimeError(spec)))
    monkeypatch.setattr(runtime, "_ort_providers_safe", lambda: ["CPUExecutionProvider"])
    report = runtime.ensure_onnxruntime("cpu")
    assert report["spec"].kind == "cpu"
    assert runtime.ensure_onnxruntime("cpu") is report
