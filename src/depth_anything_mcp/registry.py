from __future__ import annotations

import threading
from typing import Any

from depth_anything_mcp.settings import Settings, load_settings

_LOCK = threading.Lock()
_CACHE: dict[tuple[Any, ...], Any] = {}


def inference_lock() -> threading.Lock:
    return _LOCK


def cache_get(key: tuple[Any, ...]) -> Any | None:
    return _CACHE.get(key)


def cache_put(key: tuple[Any, ...], value: Any) -> Any:
    _CACHE[key] = value
    return value


def cached_models() -> list[str]:
    return [f"{key[0]}:{':'.join(str(part) for part in key[1:])}" for key in _CACHE]


def current_settings() -> Settings:
    return load_settings()
