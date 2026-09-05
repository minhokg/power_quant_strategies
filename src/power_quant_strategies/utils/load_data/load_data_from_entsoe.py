import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from power_quant_strategies.utils.load_data.EntsoeClient import EntsoeClient


def _fetch_chunk(fetch_fn: Callable[..., Any], chunk_start: pd.Timestamp, chunk_end: pd.Timestamp, kwargs: dict[str, Any]) -> tuple[int, pd.DataFrame | pd.Series | None]:
    """
    Fetch data for a single time chunk.

    Executes the provided data-fetching function for the specified time
    interval and handles missing data or other exceptions without
    interrupting the overall data retrieval process.

    :param fetch_fn: Function used to retrieve data from ENTSO-E.
    :param chunk_start: Start timestamp of the time chunk.
    :param chunk_end: End timestamp of the time chunk.
    :param kwargs: Additional keyword arguments passed to ``fetch_fn``.
    :return: A tuple containing the chunk start timestamp as an integer
        nanosecond value and the retrieved data. Returns ``None`` as the
        data value if no data is available or an error occurs.
    """
    try:
        res = fetch_fn(start=chunk_start, end=chunk_end, **kwargs)
        if res is not None and not (hasattr(res, "empty") and res.empty):
            return (chunk_start.value, res)
    except NoMatchingDataError:
        logging.warning("No data between %s and %s", chunk_start, chunk_end)
    except Exception as e:
        logging.error("Failed chunk %s to %s: %s", chunk_start, chunk_end, e)

    return (chunk_start.value, None)


def _fetch_in_chunks_parallel(fetch_fn: Callable[..., Any], start_date: pd.Timestamp, end_date: pd.Timestamp, chunk_days: int = 15, max_workers: int = 4, **kwargs: Any) -> pd.DataFrame | pd.Series:
    """
        Fetch ENTSO-E data in parallel time chunks.

    Splits the requested time range into multiple chunks and retrieves
    each chunk concurrently using a :class:`~concurrent.futures.ThreadPoolExecutor`.
    The resulting data are sorted chronologically, concatenated, and
    duplicate timestamps are removed.

    All timestamps are converted to UTC before the time range is split.

    :param fetch_fn: Function used to retrieve data from ENTSO-E.
    :param start_date: Start timestamp of the overall data retrieval period.
    :param end_date: End timestamp of the overall data retrieval period.
    :param chunk_days: Number of days included in each request chunk.
        Defaults to ``15``.
    :param max_workers: Maximum number of worker threads used for
        concurrent requests. Defaults to ``4``.
    :param kwargs: Additional keyword arguments passed to ``fetch_fn``.
    :return: A concatenated DataFrame or Series containing the retrieved
        data. Returns an empty Series if no data are retrieved.
    """
    start_utc = start_date.tz_convert("UTC") if start_date.tzinfo else start_date.tz_localize("UTC")
    end_utc = end_date.tz_convert("UTC") if end_date.tzinfo else end_date.tz_localize("UTC")

    date_intervals = pd.date_range(start=start_utc, end=end_utc, freq=f"{chunk_days}D")

    if date_intervals.empty or date_intervals[0] > start_utc:
        date_intervals = date_intervals.insert(0, start_utc)
    if date_intervals[-1] < end_utc:
        date_intervals = date_intervals.append(pd.DatetimeIndex([end_utc]))

    chunks = [(date_intervals[i], date_intervals[i + 1]) for i in range(len(date_intervals) - 1)]

    results_dict = {}

    # Fetch chunks concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_chunk, fetch_fn, c_start, c_end, kwargs) for c_start, c_end in chunks]
        for future in as_completed(futures):
            sort_key, res = future.result()
            if res is not None:
                results_dict[sort_key] = res

    if not results_dict:
        return pd.Series(dtype="float64")

    # Sort results chronologically before concatenation
    sorted_results = [results_dict[k] for k in sorted(results_dict.keys())]
    combined = pd.concat(sorted_results, axis=0)

    return combined[~combined.index.duplicated(keep="first")]


def load_data_from_entsoe(start_date: pd.Timestamp, end_date: pd.Timestamp, country_code: str = "DE_LU", max_workers_per_dataset: int = 4) -> pd.DataFrame:
    """
    Load electricity market forecasts and prices from ENTSO-E.

    Retrieves load, renewable generation, generation, day-ahead price,
    and imbalance price data for the specified bidding zone and time
    period. Each dataset is fetched in parallel using multi-day chunks
    to reduce the risk of API request failures.

    Forecast and price datasets are retrieved concurrently across
    multiple datasets. Price data are subsequently forward-filled to
    align them with the forecast data index.

    For the ``DE_LU`` bidding zone, the imbalance price is queried using
    the ``DE`` country code.

    :param start_date: Start timestamp of the data retrieval period.
    :param end_date: End timestamp of the data retrieval period.
    :param country_code: ENTSO-E bidding zone code. Defaults to
        ``"DE_LU"``.
    :param max_workers_per_dataset: Maximum number of worker threads
        used to fetch chunks for each dataset. Defaults to ``4``.
    :return: DataFrame containing load, renewable generation,
        generation forecasts, day-ahead prices, and imbalance prices,
        indexed by timestamp. Returns an empty DataFrame if no valid
        forecast data are retrieved.
    """
    logging.info("Loading ENTSO-E data for %s from %s to %s.", country_code, start_date, end_date)

    if start_date.tzinfo is None:
        start_date = start_date.tz_localize("Europe/Berlin")
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("Europe/Berlin")

    client = EntsoeClient()
    imb_country_code = "DE" if country_code == "DE_LU" else country_code

    # Map datasets to their respective query callables and arguments
    tasks = {
        "load_forecast": (client.query_load_forecast, {"country_code": country_code}),
        "wind_solar_forecast": (client.query_wind_and_solar_forecast, {"country_code": country_code}),
        "generation_forecast": (client.query_generation_forecast, {"country_code": country_code}),
        "da_price": (client.query_day_ahead_prices, {"country_code": country_code}),
        "imb_data": (client.query_imbalance_prices, {"country_code": imb_country_code}),
    }

    fetched_data = {}

    # Fetch all 5 datasets concurrently
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_key = {
            executor.submit(_fetch_in_chunks_parallel, fetch_fn, start_date=start_date, end_date=end_date, max_workers=max_workers_per_dataset, **kwargs): key
            for key, (fetch_fn, kwargs) in tasks.items()
        }

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                fetched_data[key] = future.result()
            except Exception as e:
                logging.error("Failed to fetch task %s: %s", key, e)
                fetched_data[key] = pd.Series(dtype="float64")

    # Unpack results
    load_forecast = fetched_data.get("load_forecast", pd.DataFrame())
    wind_solar_forecast = fetched_data.get("wind_solar_forecast", pd.DataFrame())
    generation_forecast = fetched_data.get("generation_forecast", pd.DataFrame())
    da_price = fetched_data.get("da_price", pd.Series(dtype="float64"))
    imb_data = fetched_data.get("imb_data", pd.Series(dtype="float64"))

    # Process Imbalance Price output
    if isinstance(imb_data, pd.DataFrame) and not imb_data.empty:
        imb_price = imb_data.iloc[:, 0].rename("imb_price")
    elif isinstance(imb_data, pd.Series) and not imb_data.empty:
        imb_price = imb_data.rename("imb_price")
    else:
        imb_price = pd.Series(dtype="float64", name="imb_price")

    if isinstance(da_price, pd.Series) and not da_price.empty:
        da_price = da_price.rename("da_price")

    # Merge forecast data
    valid_dfs = [df for df in [load_forecast, wind_solar_forecast, generation_forecast] if isinstance(df, (pd.DataFrame, pd.Series)) and not df.empty]

    if not valid_dfs:
        logging.error("No valid data retrieved.")
        return pd.DataFrame()

    data = pd.concat(valid_dfs, axis=1)

    # Reindex prices to match forecast timeframe
    data["da_price"] = da_price.reindex(data.index, method="ffill") if not da_price.empty else pd.NA
    data["imb_price"] = imb_price.reindex(data.index, method="ffill") if not imb_price.empty else pd.NA

    return data
