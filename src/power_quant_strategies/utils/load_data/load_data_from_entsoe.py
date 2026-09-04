import logging

import pandas as pd

from power_quant_strategies.utils.load_data.EntsoeClient import EntsoeClient


def load_data_from_entsoe(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    country_code: str = "DE_LU",
) -> pd.DataFrame:
    """
    Load and combine electricity market forecasts and day-ahead prices from ENTSO-E.

    The function retrieves load, wind and solar, and generation forecasts
    for the specified bidding zone and time period. Day-ahead prices are
    then aligned to the forecast data index using forward filling.

    :param start_date: Start timestamp of the data retrieval period.
    :param end_date: End timestamp of the data retrieval period.
    :param country_code: ENTSO-E bidding zone code. Defaults to `DE_LU`.
    :return: DataFrame containing load, renewable generation, generation
    forecasts, and day-ahead prices indexed by timestamp.
    """
    logging.info("Loading ENTSO-E data for %s from %s to %s.", country_code, start_date, end_date)

    client = EntsoeClient()

    # get demand forecast
    logging.info("Getting load forecast %s", country_code)
    load_forecast = client.query_load_forecast(
        country_code=country_code,
        start=start_date,
        end=end_date,
    )

    # get wind and solar generation forecast
    logging.info("Getting wind solar forecast for %s", country_code)
    wind_solar_forecast = client.query_wind_and_solar_forecast(
        country_code=country_code,
        start=start_date,
        end=end_date,
    )

    # get overall generation forecast by generation type
    logging.info("Getting wind and solar generation forecast for %s", country_code)
    generation_forecast = client.query_generation_forecast(
        country_code=country_code,
        start=start_date,
        end=end_date,
    )

    # retrieve day-ahead prices and assign a descriptive name
    logging.info("Getting Day Price for %s", country_code)
    da_price = client.query_day_ahead_prices(
        country_code=country_code,
        start=start_date,
        end=end_date,
    ).rename("da_price")

    # Imbalance Price with Fallback Handling
    logging.info("Getting imbalance prices")

    # try the target zone first, then fallback to national zone "DE"
    imb_country_code = "DE" if country_code == "DE_LU" else country_code

    logging.info("Getting imbalance prices for %s", imb_country_code)

    try:
        imb_df = client.query_imbalance_prices(
            country_code=imb_country_code,
            start=start_date,
            end=end_date,
        )

        # extract the main price column (e.g., first column or specific category)
        if isinstance(imb_df, pd.DataFrame) and not imb_df.empty:
            imb_price = imb_df.iloc[:, 0].rename("imb_price")
        elif isinstance(imb_df, pd.Series) and not imb_df.empty:
            imb_price = imb_df.rename("imb_price")
        else:
            imb_price = pd.Series(dtype="float64", name="imb_price")

    except Exception as e:
        logging.warning("Could not fetch imbalance prices for zone '%s': %s", imb_country_code, e)
        imb_price = pd.Series(dtype="float64", name="imb_price")

    # combine Base Data
    data = pd.concat([load_forecast, wind_solar_forecast, generation_forecast], axis=1)

    # align Day-Ahead Price
    data["da_price"] = da_price.reindex(data.index, method="ffill")

    # align Imbalance Price if available
    data["imb_price"] = imb_price.reindex(data.index, method="ffill")

    return data
