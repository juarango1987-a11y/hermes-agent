"""Tests for agent/browser_registry.py — provider registration & active lookup."""

from __future__ import annotations

import pytest

from agent import browser_registry
from agent.browser_provider import BrowserProvider


class _FakeProvider(BrowserProvider):
    def __init__(self, name: str, available: bool = True):
        self._name = name
        self._available = available

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def create_session(self, task_id: str):
        return {
            "session_name": f"{self._name}-session",
            "bb_session_id": f"{self._name}-id",
            "cdp_url": f"ws://{self._name}/cdp",
            "features": {},
        }

    def close_session(self, session_id: str) -> bool:
        return True

    def emergency_cleanup(self, session_id: str) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_registry():
    browser_registry._reset_for_tests()
    yield
    browser_registry._reset_for_tests()


class TestRegisterProvider:
    def test_register_and_lookup(self):
        provider = _FakeProvider("fake")
        browser_registry.register_provider(provider)
        assert browser_registry.get_provider("fake") is provider

    def test_rejects_non_provider(self):
        with pytest.raises(TypeError):
            browser_registry.register_provider("not a provider")  # type: ignore[arg-type]

    def test_register_survives_reloaded_base_class(self, monkeypatch):
        """Registry must not trust the base class frozen at module import.

        Regression for the prod warning burst
        "Failed to load plugin '<name>': register_provider() expects a
        BrowserProvider instance, got <X>": when
        ``agent.browser_provider`` is reloaded, the module-level
        ``BrowserProvider`` reference frozen in this registry diverges from
        the class real providers subclass. We simulate that divergence by
        swapping the frozen reference for an unrelated class; a genuine
        provider (subclass of the *current* canonical base) must still
        register. Pre-fix this raised TypeError; post-fix it resolves the
        base class at call time and succeeds.
        """

        class _StaleUnrelatedBase:  # stands in for a reloaded/divergent class
            pass

        monkeypatch.setattr(browser_registry, "BrowserProvider", _StaleUnrelatedBase)
        provider = _FakeProvider("reloaded")
        browser_registry.register_provider(provider)
        assert browser_registry.get_provider("reloaded") is provider

    def test_rejects_empty_name(self):
        class Empty(BrowserProvider):
            @property
            def name(self) -> str:
                return ""

            def is_available(self) -> bool:
                return True

            def create_session(self, task_id: str):
                return {}

            def close_session(self, session_id: str) -> bool:
                return True

            def emergency_cleanup(self, session_id: str) -> None:
                pass

        with pytest.raises(ValueError):
            browser_registry.register_provider(Empty())

    def test_reregister_overwrites(self):
        a = _FakeProvider("same")
        b = _FakeProvider("same")
        browser_registry.register_provider(a)
        browser_registry.register_provider(b)
        assert browser_registry.get_provider("same") is b

    def test_list_is_sorted(self):
        browser_registry.register_provider(_FakeProvider("zeta"))
        browser_registry.register_provider(_FakeProvider("alpha"))
        names = [p.name for p in browser_registry.list_providers()]
        assert names == ["alpha", "zeta"]


class TestGetActiveProvider:
    def test_single_provider_available_via_resolve(self, tmp_path, monkeypatch):
        """Test internal _resolve function with single provider."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        browser_registry.register_provider(_FakeProvider("browserbase"))
        # _resolve is internal but we can test it
        active = browser_registry._resolve(None)
        assert active is not None and active.name == "browserbase"

    def test_browserbase_preferred_on_multi_without_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        browser_registry.register_provider(_FakeProvider("browserbase"))
        browser_registry.register_provider(_FakeProvider("firecrawl"))
        active = browser_registry._resolve(None)
        assert active is not None and active.name == "browserbase"

    def test_explicit_config_wins(self, tmp_path, monkeypatch):
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"browser": {"cloud_provider": "firecrawl"}})
        )
        browser_registry.register_provider(_FakeProvider("browserbase"))
        browser_registry.register_provider(_FakeProvider("firecrawl"))
        active = browser_registry._resolve("firecrawl")
        assert active is not None and active.name == "firecrawl"

    def test_missing_configured_provider_falls_back(self, tmp_path, monkeypatch):
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"browser": {"cloud_provider": "ghost"}})
        )
        # Only browserbase is registered — configured provider doesn't exist
        browser_registry.register_provider(_FakeProvider("browserbase"))
        active = browser_registry._resolve("ghost")
        # Falls back to browserbase preference (legacy default) rather than None
        assert active is not None and active.name == "browserbase"

    def test_none_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert browser_registry._resolve(None) is None
