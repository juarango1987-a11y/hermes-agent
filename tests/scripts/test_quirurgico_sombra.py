"""Tests del modo SOMBRA silenciado del Bot Quirúrgico v2 + reporte semanal.

Cubren la tarea byc_quir_sombra_silenciar_email_dominical_20260624:
  - motor MODE=shadow NO escribe la alerta a stdout (silencio)
  - envío a James gobernado por env SEND_JAMES (default OFF)
  - enviar_telegram ya NO usa parse_mode HTML (no rompe con `<`)
  - reporte semanal agrega los .jsonl y SIEMPRE genera cuerpo en español

Los scripts viven fuera del repo (~/.hermes/scripts/byc_staffing/), se importan
vía importlib (convención SHARED_KNOWLEDGE).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytz

BYC_DIR = Path.home() / ".hermes/scripts/byc_staffing"


def _load(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, BYC_DIR / filename)
    assert spec and spec.loader, f"no se pudo cargar {filename}"
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def motor():
    return _load("quir_motor", "staffing_monitor_quirurgico.py")


@pytest.fixture(scope="module")
def reporte():
    return _load("quir_reporte", "reporte_semanal_quirurgico.py")


def _alerta_fake():
    """Alerta mínima con los campos que consumen los formateadores ES/EN.

    Incluye un `<` literal en la razón → reproduce el byte que rompía parse_mode HTML.
    """
    return {
        "puesto": "SERVER",
        "headcount": 4,
        "metric_value": 1.2,
        "p15": 3.0,
        "median": 5.0,
        "razon": "covers/server 1.2 < p15 3.0 (mediana 5.0)",
        "empleados_candidatos": [{"employeeId": "EMP-123"}],
        "ventanas_consecutivas": 2,
    }


@pytest.fixture
def motor_con_alerta(motor, monkeypatch, tmp_path):
    """Parchea el motor para que procesar_tienda dispare 1 alerta y loguee a tmp."""
    monkeypatch.setattr(motor, "DATA_DIR", tmp_path)
    monkeypatch.setattr(motor, "cargar_baseline", lambda store: {})
    monkeypatch.setattr(motor, "calcular_metricas_en_vivo",
                        lambda *a, **k: {"_meta": {"forecast": {}}})
    monkeypatch.setattr(motor, "detectar_sobra_staff",
                        lambda *a, **k: [_alerta_fake()])
    return motor


# ===== A. SILENCIO STDOUT =====
def test_shadow_no_stdout(motor_con_alerta, tmp_path, capsys, monkeypatch):
    """MODE=shadow con alerta disparada → stdout NO contiene la alerta; .jsonl SÍ la corrida."""
    monkeypatch.delenv("SEND_JAMES", raising=False)
    motor_con_alerta.procesar_tienda("BYC301", "Eastlake", "tok", {}, "shadow")

    out = capsys.readouterr().out
    assert out.strip() == "", f"stdout debería estar VACÍO, fue: {out!r}"

    log_path = tmp_path / "quirurgico_log_BYC301.jsonl"
    assert log_path.exists(), "el .jsonl de la corrida no se escribió"
    entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    assert len(entries[0]["alertas"]) == 1
    assert entries[0]["alertas"][0]["puesto"] == "SERVER"


# ===== A. JAMES GOBERNADO POR SEND_JAMES =====
def test_james_off_by_default(motor_con_alerta, monkeypatch):
    """Sin SEND_JAMES=1 → enviar_telegram NO se invoca."""
    monkeypatch.delenv("SEND_JAMES", raising=False)
    spy = MagicMock()
    monkeypatch.setattr(motor_con_alerta, "enviar_telegram", spy)

    motor_con_alerta.procesar_tienda("BYC301", "Eastlake", "tok", {}, "shadow")
    assert spy.call_count == 0, "James NO debe recibir nada por default"


def test_james_on_when_flagged(motor_con_alerta, monkeypatch):
    """Con SEND_JAMES=1 → enviar_telegram se invoca 1x al chat de James (regresión)."""
    monkeypatch.setenv("SEND_JAMES", "1")
    spy = MagicMock()
    monkeypatch.setattr(motor_con_alerta, "enviar_telegram", spy)

    motor_con_alerta.procesar_tienda("BYC301", "Eastlake", "tok", {}, "shadow")
    assert spy.call_count == 1
    assert spy.call_args.args[0] == motor_con_alerta.JAMES_CHAT_ID


# ===== A. parse_mode HTML ARREGLADO =====
def test_enviar_telegram_escapa_html(motor, monkeypatch):
    """Mensaje con `<` → payload SIN parse_mode HTML (texto plano), no provoca 400."""
    monkeypatch.setenv("BROKEN_YOKE_TELEGRAM_BOT_TOKEN", "fake-token")
    captured = {}

    class _Resp:
        status_code = 200
        text = "ok"

    def _fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(motor.requests, "post", _fake_post)

    mensaje = "Overstaffed: covers/server 1.2 < p15 3.0 <test>"
    motor.enviar_telegram(motor.JAMES_CHAT_ID, mensaje)  # no debe lanzar

    payload = captured["payload"]
    assert payload["text"] == mensaje
    assert payload.get("parse_mode") != "HTML", "parse_mode HTML reintroduce el bug del 400"
    assert "parse_mode" not in payload, "texto plano: no debe setear parse_mode"


# ===== C. REPORTE SEMANAL =====
TZ = pytz.timezone("America/Los_Angeles")


def _escribir_jsonl(data_dir: Path, store: str, lineas: list[dict]):
    p = data_dir / f"quirurgico_log_{store}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for l in lineas:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")


def test_reporte_semanal_agrega_jsonl(reporte, tmp_path):
    """Fixture con varios días/tiendas → agrega alertas por tienda/puesto; HTML no vacío en español."""
    ahora = TZ.localize(datetime(2026, 6, 24, 18, 0, 0))
    hace_2d = (ahora - timedelta(days=2)).isoformat()
    hace_1d = (ahora - timedelta(days=1)).isoformat()
    viejo = (ahora - timedelta(days=20)).isoformat()  # fuera de ventana

    _escribir_jsonl(tmp_path, "BYC301", [
        {"timestamp": hace_2d, "store": "BYC301", "alertas": [{"puesto": "SERVER"}]},
        {"timestamp": hace_1d, "store": "BYC301", "alertas": [{"puesto": "SERVER"}, {"puesto": "HOST"}]},
        {"timestamp": viejo, "store": "BYC301", "alertas": [{"puesto": "EXPO"}]},  # ignorado
    ])
    _escribir_jsonl(tmp_path, "BYC306", [
        {"timestamp": hace_1d, "store": "BYC306", "alertas": []},
        {"timestamp": hace_1d, "store": "BYC306", "error": "Telegram API falló: 400"},
    ])

    por_tienda = reporte.cargar_corridas(tmp_path, dias=7, ahora=ahora)
    agg = reporte.agregar(por_tienda)

    # BYC301: 2 corridas en ventana (el viejo se ignora), 3 alertas (SERVER x2, HOST x1)
    assert agg["por_tienda"]["BYC301"]["corridas"] == 2
    assert agg["por_tienda"]["BYC301"]["alertas"] == 3
    assert agg["por_tienda"]["BYC301"]["por_puesto"]["SERVER"] == 2
    assert agg["por_tienda"]["BYC301"]["por_puesto"]["HOST"] == 1
    # BYC306: 2 corridas, 0 alertas, 1 error
    assert agg["por_tienda"]["BYC306"]["errores"] == 1
    assert agg["total_alertas"] == 3
    assert agg["por_puesto_global"]["SERVER"] == 2

    html = reporte.construir_html(agg, dias=7, ahora=ahora)
    assert html.strip()
    assert "Resumen semanal" in html
    assert "SERVER" in html
    assert "Telegram API falló" in html  # error del motor listado


def test_reporte_envia_siempre(reporte, tmp_path):
    """Semana con 0 alertas → HTML igual se genera y lo dice explícito."""
    ahora = TZ.localize(datetime(2026, 6, 24, 18, 0, 0))
    _escribir_jsonl(tmp_path, "BYC301", [
        {"timestamp": (ahora - timedelta(days=1)).isoformat(), "store": "BYC301", "alertas": []},
    ])

    por_tienda = reporte.cargar_corridas(tmp_path, dias=7, ahora=ahora)
    agg = reporte.agregar(por_tienda)
    assert agg["total_alertas"] == 0
    assert agg["total_corridas"] == 1

    html = reporte.construir_html(agg, dias=7, ahora=ahora)
    assert html.strip(), "el reporte NUNCA debe quedar vacío"
    assert "0 alertas" in html
