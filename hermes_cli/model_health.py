"""Model health tracking and periodic validation helpers.

This module provides:
- Last-known-working model tracking (for runtime fallback).
- Best-effort periodic health checks for configured models.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_HEALTH_PATH = get_hermes_home() / "model_health.json"
_LOCK = threading.Lock()


def _now_ts() -> float:
    return time.time()


def _load() -> Dict[str, Any]:
    try:
        if not _HEALTH_PATH.exists():
            return {"models": {}, "last_health_check_at": 0.0, "history": []}
        return json.loads(_HEALTH_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"models": {}, "last_health_check_at": 0.0, "history": []}


def _save(data: Dict[str, Any]) -> None:
    _HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HEALTH_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_key(provider: str, model: str, base_url: str = "") -> str:
    p = (provider or "").strip().lower()
    m = (model or "").strip()
    b = (base_url or "").strip().rstrip("/")
    return f"{p}|{m}|{b}"


def record_model_success(provider: str, model: str, base_url: str = "") -> None:
    """Record a successful inference call for a model route."""
    if not provider or not model:
        return
    ts = _now_ts()
    key = _model_key(provider, model, base_url)
    with _LOCK:
        data = _load()
        models = data.setdefault("models", {})
        row = models.setdefault(key, {})
        row.update(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url or "",
                "last_success_at": ts,
                "last_status": "ok",
                "last_error": "",
            }
        )
        _save(data)


def record_model_failure(
    provider: str,
    model: str,
    base_url: str = "",
    *,
    error: str = "",
) -> None:
    """Record a failed inference attempt for a model route."""
    if not provider or not model:
        return
    ts = _now_ts()
    key = _model_key(provider, model, base_url)
    with _LOCK:
        data = _load()
        models = data.setdefault("models", {})
        row = models.setdefault(key, {})
        row.update(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url or "",
                "last_failure_at": ts,
                "last_status": "fail",
                "last_error": (error or "")[:300],
            }
        )
        _save(data)


def get_last_known_working_model(
    *,
    exclude_provider: str = "",
    exclude_model: str = "",
    exclude_base_url: str = "",
) -> Optional[Dict[str, str]]:
    """Return the most recently successful model route, if any."""
    with _LOCK:
        data = _load()
    models = data.get("models", {}) or {}
    rows = list(models.values())
    if not rows:
        return None

    ex_p = (exclude_provider or "").strip().lower()
    ex_m = (exclude_model or "").strip()
    ex_b = (exclude_base_url or "").strip().rstrip("/")

    best = None
    best_ts = -1.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = float(row.get("last_success_at") or 0.0)
        if ts <= 0:
            continue
        p = str(row.get("provider", "")).strip().lower()
        m = str(row.get("model", "")).strip()
        b = str(row.get("base_url", "")).strip().rstrip("/")
        if p == ex_p and m == ex_m and b == ex_b:
            continue
        if ts > best_ts:
            best_ts = ts
            best = {"provider": p, "model": m, "base_url": b}
    return best


def _iter_configured_models(config: Dict[str, Any]) -> Iterable[Dict[str, str]]:
    seen: set[str] = set()

    def _yield(provider: str, model: str, base_url: str = "", api_key: str = ""):
        p = (provider or "").strip()
        m = (model or "").strip()
        b = (base_url or "").strip()
        if not p or not m:
            return
        key = _model_key(p, m, b)
        if key in seen:
            return
        seen.add(key)
        yield {"provider": p, "model": m, "base_url": b, "api_key": api_key or ""}

    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    if isinstance(model_cfg, dict):
        p = model_cfg.get("provider", "")
        m = model_cfg.get("default", "") or model_cfg.get("model", "")
        b = model_cfg.get("base_url", "")
        if p and m:
            yield from _yield(p, m, b)

    fb = config.get("fallback_providers") or config.get("fallback_model") or []
    if isinstance(fb, dict):
        fb = [fb]
    if isinstance(fb, list):
        for row in fb:
            if not isinstance(row, dict):
                continue
            yield from _yield(
                row.get("provider", ""),
                row.get("model", ""),
                row.get("base_url", ""),
                row.get("api_key", ""),
            )

    # legacy + compatibility path for user endpoints
    try:
        from hermes_cli.config import get_compatible_custom_providers

        cp = get_compatible_custom_providers(config)
    except Exception:
        cp = config.get("custom_providers", [])
    if isinstance(cp, list):
        for row in cp:
            if not isinstance(row, dict):
                continue
            name = row.get("name", "")
            provider = f"custom:{name}" if name else "custom"
            yield from _yield(
                provider,
                row.get("model", ""),
                row.get("base_url", "") or row.get("url", "") or row.get("api", ""),
                row.get("api_key", ""),
            )


def run_periodic_health_check(config: Dict[str, Any], *, timeout: float = 5.0) -> Dict[str, Any]:
    """Validate all configured models and persist health status."""
    from hermes_cli.models import validate_requested_model

    now_ts = _now_ts()
    results: list[dict] = []
    for item in _iter_configured_models(config):
        provider = item["provider"]
        model = item["model"]
        base_url = item.get("base_url", "")
        api_key = item.get("api_key", "")
        try:
            verdict = validate_requested_model(
                model,
                provider,
                api_key=api_key,
                base_url=base_url,
                strict=True,
            )
            ok = bool(verdict.get("accepted"))
            msg = verdict.get("message") or ""
        except Exception as exc:
            ok = False
            msg = str(exc)

        if ok:
            record_model_success(provider, model, base_url)
        else:
            record_model_failure(provider, model, base_url, error=msg)

        results.append(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "ok": ok,
                "message": msg,
            }
        )

    with _LOCK:
        data = _load()
        data["last_health_check_at"] = now_ts
        hist = data.setdefault("history", [])
        hist.append({"ts": now_ts, "results": results})
        if len(hist) > 20:
            data["history"] = hist[-20:]
        _save(data)

    return {"checked": len(results), "results": results}


def maybe_start_periodic_health_check_async(config: Dict[str, Any]) -> bool:
    """Run periodic model health checks in a background daemon thread when due."""
    if not isinstance(config, dict):
        return False
    mh = config.get("model_health", {}) or {}
    if not bool(mh.get("enabled", True)):
        return False

    interval_h = mh.get("check_interval_hours", 24)
    try:
        interval_s = max(3600, int(float(interval_h) * 3600))
    except (TypeError, ValueError):
        interval_s = 24 * 3600

    with _LOCK:
        data = _load()
        last = float(data.get("last_health_check_at") or 0.0)
    if last > 0 and (_now_ts() - last) < interval_s:
        return False

    def _runner() -> None:
        try:
            run_periodic_health_check(config)
        except Exception as exc:
            logger.debug("Periodic model health check failed: %s", exc)

    threading.Thread(target=_runner, daemon=True, name="model-health-check").start()
    return True
