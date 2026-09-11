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


def test_ensure_onnxruntime_cpu_skips_install(monkeypatch) -> None:
    from depth_anything_mcp import runtime

    runtime._ENSURED.clear()
    monkeypatch.setattr(runtime, "_install_wheel", lambda spec: (_ for _ in ()).throw(RuntimeError(spec)))
    monkeypatch.setattr(runtime, "_ort_providers_safe", lambda: ["CPUExecutionProvider"])
    report = runtime.ensure_onnxruntime("cpu")
    assert report["spec"].kind == "cpu"
    assert runtime.ensure_onnxruntime("cpu") is report
