from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from power_quant_strategies.utils.load_data.load_data_from_open_meteo import load_data_from_open_meteo

HOURLY = {"time": ["2026-01-01T00:00", "2026-01-01T01:00"], "temperature_2m": [1.5, 2.0], "wind_speed_10m": [3.0, 4.0]}


def _response(payload=None):
    response = Mock()
    response.json.return_value = {"hourly": payload or HOURLY}
    response.raise_for_status.return_value = None
    return response


@patch("power_quant_strategies.utils.load_data.load_data_from_open_meteo.requests.get")
def test_historic_dates_use_the_archive_api(mock_get):
    mock_get.return_value = _response()

    data = load_data_from_open_meteo(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), latitude=48.1, longitude=11.6, timeout=5)

    url = mock_get.call_args.args[0]
    params = mock_get.call_args.kwargs["params"]
    assert url == "https://archive-api.open-meteo.com/v1/archive"
    assert (params["start_date"], params["end_date"]) == ("2026-01-01", "2026-01-02")
    assert (params["latitude"], params["longitude"]) == (48.1, 11.6)
    assert mock_get.call_args.kwargs["timeout"] == 5

    assert list(data.columns) == ["temperature_2m", "wind_speed_10m"]
    assert str(data.index.tz) == "Europe/Berlin"
    assert data.loc[pd.Timestamp("2026-01-01 01:00", tz="Europe/Berlin"), "temperature_2m"] == 2.0


@patch("power_quant_strategies.utils.load_data.load_data_from_open_meteo.requests.get")
def test_recent_dates_use_the_forecast_api(mock_get):
    mock_get.return_value = _response()
    today = pd.Timestamp.now().floor("D")

    load_data_from_open_meteo(today - pd.Timedelta(days=1), today + pd.Timedelta(days=2))

    assert mock_get.call_args.args[0] == "https://api.open-meteo.com/v1/forecast"


@patch("power_quant_strategies.utils.load_data.load_data_from_open_meteo.requests.get")
def test_timezone_aware_dates_are_formatted_as_local_dates(mock_get):
    mock_get.return_value = _response()
    load_data_from_open_meteo(pd.Timestamp("2026-01-01 23:00", tz="Europe/Berlin"), pd.Timestamp("2026-01-03", tz="Europe/Berlin"))
    assert mock_get.call_args.kwargs["params"]["start_date"] == "2026-01-01"


@patch("power_quant_strategies.utils.load_data.load_data_from_open_meteo.requests.get")
def test_http_errors_are_raised(mock_get):
    response = _response()
    response.raise_for_status.side_effect = requests.HTTPError("400")
    mock_get.return_value = response
    with pytest.raises(requests.HTTPError):
        load_data_from_open_meteo(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))
