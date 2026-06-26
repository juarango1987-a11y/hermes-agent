"""Tests for agent/web_search_registry.py — provider registration & active lookup."""

from __future__ import annotations

import pytest

from agent import web_search_registry
from agent.web_search_provider import WebSearchProvider


class _FakeProvider(WebSearchProvider):
    def __init__(self, name: str, available: bool = True, search: bool = True, extract: bool = True):
        self._name = name
        self._available = available
        self._search = search
        self._extract = extract

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def supports_search(self) -> bool:
        return self._search

    def supports_extract(self) -> bool:
        return self._extract

    def search(self, query: str, limit: int = 5):
        return {"success": True, "data": {"web": [{"title": f"{self._name}://{query}", "url": "http://fake", "description": "fake", "position": 1}]}}

    def extract(self, urls, **kwargs):
        return {"success": True, "data": [{"url": u, "title": f"{self._name}", "content": f"content from {u}", "raw_content": "", "metadata": {}} for u in urls]}


@pytest.fixture(autouse=True)
def _reset_registry():
    web_search_registry._reset_for_tests()
    yield
    web_search_registry._reset_for_tests()


class TestRegisterProvider:
    def test_register_and_lookup(self):
        provider = _FakeProvider("fake")
        web_search_registry.register_provider(provider)
        assert web_search_registry.get_provider("fake") is provider

    def test_rejects_non_provider(self):
        with pytest.raises(TypeError):
            web_search_registry.register_provider("not a provider")  # type: ignore[arg-type]

    def test_register_survives_reloaded_base_class(self, monkeypatch):
        """Registry must not trust the base class frozen at module import.

        Regression for the prod warning burst
        "Failed to load plugin '<name>': register_provider() expects a
        WebSearchProvider instance, got <X>": when
        ``agent.web_search_provider`` is reloaded, the module-level
        ``WebSearchProvider`` reference frozen in this registry diverges from
        the class real providers subclass. We simulate that divergence by
        swapping the frozen reference for an unrelated class; a genuine
        provider (subclass of the *current* canonical base) must still
        register. Pre-fix this raised TypeError; post-fix it resolves the
        base class at call time and succeeds.
        """

        class _StaleUnrelatedBase:  # stands in for a reloaded/divergent class
            pass

        monkeypatch.setattr(web_search_registry, "WebSearchProvider", _StaleUnrelatedBase)
        provider = _FakeProvider("reloaded")
        web_search_registry.register_provider(provider)
        assert web_search_registry.get_provider("reloaded") is provider

    def test_rejects_empty_name(self):
        class Empty(WebSearchProvider):
            @property
            def name(self) -> str:
                return ""

            def is_available(self) -> bool:
                return True

            def search(self, query: str, limit: int = 5):
                return {}

        with pytest.raises(ValueError):
            web_search_registry.register_provider(Empty())

    def test_reregister_overwrites(self):
        a = _FakeProvider("same")
        b = _FakeProvider("same")
        web_search_registry.register_provider(a)
        web_search_registry.register_provider(b)
        assert web_search_registry.get_provider("same") is b

    def test_list_is_sorted(self):
        web_search_registry.register_provider(_FakeProvider("zeta"))
        web_search_registry.register_provider(_FakeProvider("alpha"))
        names = [p.name for p in web_search_registry.list_providers()]
        assert names == ["alpha", "zeta"]


class TestGetActiveSearchProvider:
    def test_single_eligible_search_provider_autoresolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        web_search_registry.register_provider(_FakeProvider("solo", search=True, extract=False))
        active = web_search_registry.get_active_search_provider()
        assert active is not None and active.name == "solo"

    def test_explicit_config_wins(self, tmp_path, monkeypatch):
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"web": {"search_backend": "tavily"}})
        )
        web_search_registry.register_provider(_FakeProvider("tavily"))
        web_search_registry.register_provider(_FakeProvider("exa"))
        active = web_search_registry.get_active_search_provider()
        assert active is not None and active.name == "tavily"

    def test_legacy_preference_order(self, tmp_path, monkeypatch):
        """When no config is set, prefer firecrawl > parallel > tavily > exa."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        web_search_registry.register_provider(_FakeProvider("exa"))
        web_search_registry.register_provider(_FakeProvider("tavily"))
        active = web_search_registry.get_active_search_provider()
        # Tavily comes before exa in legacy preference
        assert active is not None and active.name == "tavily"

    def test_none_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert web_search_registry.get_active_search_provider() is None


class TestGetActiveExtractProvider:
    def test_single_eligible_extract_provider_autoresolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        web_search_registry.register_provider(_FakeProvider("solo", search=False, extract=True))
        active = web_search_registry.get_active_extract_provider()
        assert active is not None and active.name == "solo"

    def test_explicit_config_wins(self, tmp_path, monkeypatch):
        import yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"web": {"extract_backend": "firecrawl"}})
        )
        web_search_registry.register_provider(_FakeProvider("firecrawl"))
        web_search_registry.register_provider(_FakeProvider("parallel"))
        active = web_search_registry.get_active_extract_provider()
        assert active is not None and active.name == "firecrawl"

    def test_none_when_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert web_search_registry.get_active_extract_provider() is None
