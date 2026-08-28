import pandas as pd

from power_quant_strategies.utils.load_data.EntsoeClient import EntsoeClient


def load_data_from_entsoe(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    country: str = "DE_LU",
) -> pd.DataFrame:
    """
    Load and combine electricity market forecasts and day-ahead prices from ENTSO-E.

    The function retrieves load, wind and solar, and generation forecasts
    for the specified bidding zone and time period. Day-ahead prices are
    then aligned to the forecast data index using forward filling.

    :param start_date: Start timestamp of the data retrieval period.
    :param end_date: End timestamp of the data retrieval period.
    :param country: ENTSO-E bidding zone code. Defaults to `DE_LU`.
    :return: DataFrame containing load, renewable generation, generation
    forecasts, and day-ahead prices indexed by timestamp.
    """
    client = EntsoeClient()

    # get demand forecast
    load_forecast = client.query_load_forecast(
        country_code=country,
        start=start_date,
        end=end_date,
    )

    # get wind and solar generation forecast
    wind_solar_forecast = client.query_wind_and_solar_forecast(
        country_code=country,
        start=start_date,
        end=end_date,
    )

    # get overall generation forecast by generation type
    generation_forecast = client.query_generation_forecast(
        country_code=country,
        start=start_date,
        end=end_date,
    )

    # retrieve day-ahead prices and assign a descriptive name
    da_price = client.query_day_ahead_prices(
        country_code=country,
        start=start_date,
        end=end_date,
    ).rename("da_price")

    # combine all forecast data using their timestamps as the index
    data = pd.concat(
        [
            load_forecast,
            wind_solar_forecast,
            generation_forecast,
        ],
        axis=1,
    )

    # align day-ahead prices with the forecast timestamps
    # forward filling handles differences in frequency or timestamp coverage
    da_price = da_price.reindex(data.index, method="ffill")

    data["da_price"] = da_price

    return data
