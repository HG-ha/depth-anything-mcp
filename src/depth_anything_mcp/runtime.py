from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass

ORT_PACKAGES = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
# 1.21-1.26 default GPU wheels target CUDA 12.x. 1.27+ switched to CUDA 13.
CUDA12_WHEEL = "onnxruntime-gpu[cuda,cudnn]>=1.21.0,<1.27"
DIRECTML_WHEEL = "onnxruntime-directml>=1.18.0"
CPU_WHEEL = "onnxruntime>=1.18.0"

_ENSURED: dict[str, dict] = {}


@dataclass(frozen=True)
class DeviceSpec:
    kind: str
    device_id: int = 0

    @property
    def label(self) -> str:
        if self.kind == "cuda":
            return f"cuda:{self.device_id}"
        return self.kind


def is_windows() -> bool:
    return sys.platform == "win32"


def parse_device_spec(value: str | None) -> DeviceSpec:
    text = (value or "auto").strip().lower()
    if text in {"", "auto"}:
        return DeviceSpec("auto")
    if text in {"cpu"}:
        return DeviceSpec("cpu")
    if text in {"dml", "directml"}:
        return DeviceSpec("dml")
    if text in {"gpu", "cuda"}:
        return DeviceSpec("cuda", 0)
    if text.startswith("cuda:"):
        return DeviceSpec("cuda", int(text.split(":", 1)[1]))
    if text.isdigit():
        return DeviceSpec("cuda", int(text))
    raise ValueError("device must be auto, cpu, dml, cuda, or cuda:N")


def has_nvidia() -> bool:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        completed = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return False
    return completed.returncode == 0 and "GPU" in (completed.stdout or "")


def resolve_device_spec(value: str | None) -> DeviceSpec:
    spec = parse_device_spec(value)
    if spec.kind != "auto":
        return spec
    if is_windows():
        return DeviceSpec("dml")
    if has_nvidia():
        return DeviceSpec("cuda", 0)
    return DeviceSpec("cpu")


def _log(message: str) -> None:
    print(f"[depth-anything] {message}", file=sys.stderr, flush=True)


def _purge_ort_modules() -> None:
    for name in list(sys.modules):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            del sys.modules[name]


def _ort_providers() -> list[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def _ort_providers_safe() -> list[str]:
    try:
        return _ort_providers()
    except Exception:
        return []


def _has_ep(name: str) -> bool:
    return name in _ort_providers_safe()


def _run_install(args: list[str]) -> None:
    commands: list[list[str]] = [
        [sys.executable, "-m", "pip", "--disable-pip-version-check", *args],
    ]
    uv = shutil.which("uv")
    if uv:
        if args and args[0] == "install":
            commands.append([uv, "pip", "install", "--python", sys.executable, *args[1:]])
        elif args and args[0] == "uninstall":
            commands.append([uv, "pip", "uninstall", "--python", sys.executable, *args[1:]])
    errors: list[str] = []
    for command in commands:
        try:
            subprocess.check_call(command, stdout=sys.stderr, timeout=600)
            return
        except Exception as exc:
            errors.append(f"{' '.join(command)} -> {exc}")
    raise RuntimeError(" ; ".join(errors) or "package install failed")


def _install_wheel(spec: str) -> None:
    try:
        _run_install(["uninstall", "-y", *ORT_PACKAGES])
        _run_install(["install", spec])
        _purge_ort_modules()
    except Exception:
        try:
            _run_install(["install", CPU_WHEEL])
            _purge_ort_modules()
        except Exception:
            pass
        raise


def _wanted_ep(kind: str) -> str | None:
    if kind == "cuda":
        return "CUDAExecutionProvider"
    if kind == "dml":
        return "DmlExecutionProvider"
    return None


def ensure_onnxruntime(preferred: str | None = None) -> dict:
    """Pick a GPU runtime without asking the user to match CUDA versions.

    Windows: onnxruntime-directml (NVIDIA / AMD / Intel).
    Linux + NVIDIA: onnxruntime-gpu pinned to CUDA 12 (ORT < 1.27).
    Anything else, or any failure: CPU.
    """
    cache_key = (preferred or "auto").strip().lower() or "auto"
    cached = _ENSURED.get(cache_key)
    if cached is not None:
        return cached

    requested = parse_device_spec(preferred)
    resolved = resolve_device_spec(preferred)
    report: dict = {
        "requested": requested.label,
        "nvidia": has_nvidia(),
        "windows": is_windows(),
        "providers": [],
        "spec": DeviceSpec("cpu"),
    }

    if resolved.kind == "dml" and not is_windows():
        _log("DirectML is Windows-only; using CPU.")
        report["providers"] = _ort_providers_safe()
        _ENSURED[cache_key] = report
        return report

    if resolved.kind == "cpu":
        report["providers"] = _ort_providers_safe()
        _ENSURED[cache_key] = report
        return report

    wanted = _wanted_ep(resolved.kind)
    if wanted and _has_ep(wanted):
        report["providers"] = _ort_providers_safe()
        report["spec"] = resolved
        _ENSURED[cache_key] = report
        return report

    try:
        if resolved.kind == "dml":
            _log("Installing onnxruntime-directml (no CUDA version pin).")
            _install_wheel(DIRECTML_WHEEL)
        else:
            _log("Installing onnxruntime-gpu for CUDA 12 (ORT 1.21-1.26, plus cuda/cudnn wheels).")
            _install_wheel(CUDA12_WHEEL)

        if wanted and _has_ep(wanted):
            report["providers"] = _ort_providers_safe()
            report["spec"] = resolved
            _log(f"GPU runtime ready: {resolved.label}")
            _ENSURED[cache_key] = report
            return report

        if resolved.kind == "cuda" and is_windows():
            _log("CUDA EP unavailable; trying DirectML instead.")
            _install_wheel(DIRECTML_WHEEL)
            if _has_ep("DmlExecutionProvider"):
                report["providers"] = _ort_providers_safe()
                report["spec"] = DeviceSpec("dml")
                report["fallback"] = "dml"
                _log("DirectML is ready.")
                _ENSURED[cache_key] = report
                return report

        _log("GPU execution provider unavailable; using CPU.")
    except Exception as exc:
        report["error"] = str(exc)
        _log(f"GPU runtime install failed, using CPU. {exc}")

    report["providers"] = _ort_providers_safe()
    report["spec"] = DeviceSpec("cpu")
    _ENSURED[cache_key] = report
    return report


def ort_providers_for(spec: DeviceSpec) -> list:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if spec.kind == "cuda" and "CUDAExecutionProvider" in available:
        return [
            ("CUDAExecutionProvider", {"device_id": int(spec.device_id)}),
            "CPUExecutionProvider",
        ]
    if spec.kind == "dml" and "DmlExecutionProvider" in available:
        return [
            ("DmlExecutionProvider", {"device_id": int(spec.device_id)}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]
