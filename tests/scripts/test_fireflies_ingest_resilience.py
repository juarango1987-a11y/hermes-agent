"""Tests de resiliencia HTTP para el cron summarizer (fireflies_ingest.py).

El script vive fuera del repo (perfil summarizer); se importa vía importlib
según la convención de SHARED_KNOWLEDGE. Mockeamos urllib + time.sleep para no
tocar red ni esperar de verdad.
"""
from __future__ import annotations

import importlib.util
import io
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

SCRIPT = (
    Path.home()
    / ".hermes/profiles/summarizer/scripts/fireflies_ingest.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("fireflies_ingest_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ff = _load()


def _http_error(code: int, body: bytes = b"boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, f"status {code}", {}, io.BytesIO(body))


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(payload: dict):
    import json

    return _FakeResp(json.dumps(payload).encode())


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Backoff instantáneo y jitter determinista.
    monkeypatch.setattr(ff.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ff.random, "uniform", lambda *_: 0.0)


# ── http_json ──────────────────────────────────────────────────────────────

def test_retry_503_then_success(monkeypatch):
    """T1: dos 503 seguidos y luego 200 → devuelve data en el 3er intento."""
    seq = [_http_error(503), _http_error(503), _ok({"ok": 1})]
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        r = seq[i]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    out = ff.http_json("http://x", {}, {}, timeout=5, phase="test")
    assert out == {"ok": 1}
    assert calls["n"] == 3


def test_persistent_503_raises_transient(monkeypatch):
    """T2: 503 persistente → TransientError tras MAX_RETRIES+1 intentos."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(503)

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ff.TransientError):
        ff.http_json("http://x", {}, {}, timeout=5, phase="test")
    assert calls["n"] == ff.MAX_RETRIES + 1


def test_401_is_permanent_no_retry(monkeypatch):
    """T3: 401 → PermanentError inmediato, sin reintentar."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ff.PermanentError):
        ff.http_json("http://x", {}, {}, timeout=5, phase="test")
    assert calls["n"] == 1


def test_429_is_retryable(monkeypatch):
    """T4: 429 se reintenta (rate limit)."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(429)

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ff.TransientError):
        ff.http_json("http://x", {}, {}, timeout=5, phase="test")
    assert calls["n"] == ff.MAX_RETRIES + 1


def test_urlerror_is_retryable(monkeypatch):
    """URLError (red/timeout) → transitorio, se reintenta."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ff.TransientError):
        ff.http_json("http://x", {}, {}, timeout=5, phase="test")
    assert calls["n"] == ff.MAX_RETRIES + 1


def test_backoff_grows_exponentially(monkeypatch):
    """T5: los delays crecen 2,4,8 (BASE_DELAY=2, jitter=0)."""
    delays = []
    monkeypatch.setattr(ff.time, "sleep", lambda d: delays.append(d))
    monkeypatch.setattr(ff.random, "uniform", lambda *_: 0.0)
    monkeypatch.setattr(ff.urllib.request, "urlopen",
                        mock.Mock(side_effect=_http_error(503)))
    with pytest.raises(ff.TransientError):
        ff.http_json("http://x", {}, {}, timeout=5, phase="test")
    assert delays == [ff.BASE_DELAY * 2 ** i for i in range(ff.MAX_RETRIES)]
    assert delays == sorted(delays) and delays[0] < delays[-1]


# ── gql clasificación ────────────────────────────────────────────────────────

def test_gql_graphql_errors_are_permanent(monkeypatch):
    """T6: respuesta 200 con 'errors' de GraphQL → PermanentError (no reintenta)."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _ok({"errors": [{"message": "bad query"}]})

    monkeypatch.setattr(ff.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ff.PermanentError):
        ff.gql("{ x }")
    assert calls["n"] == 1


# ── main: circuit breaker + exit code ────────────────────────────────────────

def _stub_main_env(monkeypatch, meetings):
    monkeypatch.setattr(ff, "FIREFLIES_KEY", "k")
    monkeypatch.setattr(ff, "GEMINI_KEY", "k")
    monkeypatch.setattr(ff, "SOUL_PATH", Path("/nonexistent/soul.md"))
    monkeypatch.setattr(ff, "list_recent", lambda h: meetings)
    monkeypatch.setattr(ff, "already_processed", lambda d, t: None)
    monkeypatch.setattr(ff, "render_transcript", lambda s: "t")
    monkeypatch.setattr(ff, "coo_analyze", lambda *a: "análisis")
    monkeypatch.setattr(ff, "write_note",
                        lambda *a: Path(f"/tmp/{ff.slugify(a[1])}.md"))


def test_circuit_breaker_trips_and_skips(monkeypatch, capsys):
    """T7: fetch siempre transitorio → tras BREAKER_THRESHOLD salta el resto; exit≠0."""
    meetings = [{"id": str(i), "title": f"M{i}", "date": "2026-07-01T10:00:00"}
                for i in range(6)]
    _stub_main_env(monkeypatch, meetings)

    def boom(mid):
        raise ff.TransientError("[fireflies] HTTP 503 — persistió")

    monkeypatch.setattr(ff, "fetch_full", boom)
    rc = ff.main()
    out = capsys.readouterr().out
    assert rc == 1  # cero procesadas + errores → fallo total → alerta
    assert "circuit-breaker" in out or "servicio degradado" in out.lower()


def test_partial_failure_exits_zero(monkeypatch, capsys):
    """T8: 1 OK + 1 transitorio → exit 0 (parcial), errores reportados en texto."""
    meetings = [{"id": "1", "title": "Buena", "date": "2026-07-01T10:00:00"},
                {"id": "2", "title": "Mala", "date": "2026-07-01T11:00:00"}]
    _stub_main_env(monkeypatch, meetings)

    def fetch(mid):
        if mid == "2":
            raise ff.TransientError("[gemini] HTTP 503 — persistió")
        return {"sentences": [], "summary": {}}

    monkeypatch.setattr(ff, "fetch_full", fetch)
    rc = ff.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 junta(s) analizada(s)" in out
    assert "Mala" in out and "transitorio" in out


def test_total_failure_exits_nonzero(monkeypatch, capsys):
    """Todas fallan (permanente) → exit≠0 para disparar alerta Kuma."""
    meetings = [{"id": "1", "title": "Uno", "date": "2026-07-01T10:00:00"}]
    _stub_main_env(monkeypatch, meetings)

    def boom(mid):
        raise ff.PermanentError("[gemini] HTTP 401 (permanente)")

    monkeypatch.setattr(ff, "fetch_full", boom)
    rc = ff.main()
    assert rc == 1
    assert "permanente" in capsys.readouterr().out
