import pytest

from power_quant_strategies.utils.load_data.entsoe_client import EntsoeClient


def test_needs_api_key(monkeypatch):
    monkeypatch.delenv("ENTSOE_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ENTSOE_API_KEY"):
        EntsoeClient()


def test_uses_api_key_and_retries(monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "dummy-key")

    client = EntsoeClient()

    assert client.api_key == "dummy-key"
    assert client.timeout == 300
    adapter = client.session.get_adapter("https://web-api.tp.entsoe.eu")
    assert adapter.max_retries.total == 5
    assert 429 in adapter.max_retries.status_forcelist
