import pandas as pd
import requests


def load_data_from_open_meteo(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    latitude: float = 52.5200,
    longitude: float = 13.4050,
    timezone: str = "Europe/Berlin",
    timeout: float = 10,
) -> pd.DataFrame:
    """
    Fetch hourly weather forecast data from the Open-Meteo API.

    :param start_date: Start timestamp for data retrieval.
    :param end_date: End timestamp for data retrieval.
    :param latitude: Target latitude coordinate, defaults to 52.5200.
    :param longitude: Target longitude coordinate, defaults to 13.4050.
    :param timezone: Target timezone for data, defaults to "Europe/Berlin".
    :param timeout: Timeout for data retrieval, defaults to 10.

    :return: DataFrame containing weather features indexed by timezone-aware
      pd.Timestamp.
    """
    # 1. Normalize dates for Open-Meteo string formatting
    start_dt = pd.Timestamp(start_date).tz_localize(None) if pd.Timestamp(start_date).tzinfo else pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date).tz_localize(None) if pd.Timestamp(end_date).tzinfo else pd.Timestamp(end_date)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # 2. Automatically select Archive API for historical dates
    today = pd.Timestamp.now().floor("D")
    if start_dt < (today - pd.Timedelta(days=7)):
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_str,
        "end_date": end_str,
        "timezone": timezone,
        "hourly": [
            "temperature_2m",
            "direct_normal_irradiance",
            "global_tilted_irradiance",
            "wind_speed_10m",
            "wind_speed_100m",
            "cloud_cover",
        ],
    }

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    # 3. Build DataFrame and set timezone-aware DatetimeIndex
    raw_json = response.json()["hourly"]
    df = pd.DataFrame(raw_json)

    # Convert ISO strings to timezone-aware timestamps matching target timezone
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    df.set_index("time", inplace=True)

    return df
