from unittest.mock import Mock, patch

import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from power_quant_strategies.utils.load_data.load_data_from_entsoe import _fetch_chunk, _fetch_in_chunks_parallel, load_data_from_entsoe


class TestFetchChunk:
    """Tests for _fetch_chunk."""

    def test_returns_data_when_fetch_succeeds(self):
        """Return fetched data when the API call succeeds."""
        fetch_fn = Mock()
        expected = pd.DataFrame({"load": [100, 200]})
        fetch_fn.return_value = expected

        chunk_start = pd.Timestamp("2026-01-01", tz="UTC")
        chunk_end = pd.Timestamp("2026-01-02", tz="UTC")

        result = _fetch_chunk(
            fetch_fn=fetch_fn,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            kwargs={"country_code": "DE_LU"},
        )

        assert result[0] == chunk_start.value
        pd.testing.assert_frame_equal(result[1], expected)

        fetch_fn.assert_called_once_with(
            start=chunk_start,
            end=chunk_end,
            country_code="DE_LU",
        )

    def test_returns_none_when_result_is_empty(self):
        """Return None when the API returns an empty DataFrame."""
        fetch_fn = Mock()
        fetch_fn.return_value = pd.DataFrame()

        chunk_start = pd.Timestamp("2026-01-01", tz="UTC")
        chunk_end = pd.Timestamp("2026-01-02", tz="UTC")

        result = _fetch_chunk(
            fetch_fn=fetch_fn,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            kwargs={},
        )

        assert result == (chunk_start.value, None)

    def test_handles_no_matching_data_error(self, caplog):
        """Handle NoMatchingDataError without raising."""
        fetch_fn = Mock()
        fetch_fn.side_effect = NoMatchingDataError()

        chunk_start = pd.Timestamp("2026-01-01", tz="UTC")
        chunk_end = pd.Timestamp("2026-01-02", tz="UTC")

        result = _fetch_chunk(
            fetch_fn=fetch_fn,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            kwargs={},
        )

        assert result == (chunk_start.value, None)
        assert "No data between" in caplog.text

    def test_handles_unexpected_exceptions(self, caplog):
        """Handle NoMatchingDataError without raising."""
        fetch_fn = Mock()
        fetch_fn.side_effect = NoMatchingDataError()

        chunk_start = pd.Timestamp("2026-01-01", tz="UTC")
        chunk_end = pd.Timestamp("2026-01-02", tz="UTC")

        result = _fetch_chunk(fetch_fn=fetch_fn, chunk_start=chunk_start, chunk_end=chunk_end, kwargs={})

        assert result == (chunk_start.value, None)
        assert "No data between" in caplog.text

    def test_handles_unexpected_exception(self, caplog):
        """Handle unexpected exceptions without raising."""
        fetch_fn = Mock()
        fetch_fn.side_effect = RuntimeError("API failure")

        chunk_start = pd.Timestamp("2026-01-01", tz="UTC")
        chunk_end = pd.Timestamp("2026-01-02", tz="UTC")

        result = _fetch_chunk(fetch_fn=fetch_fn, chunk_start=chunk_start, chunk_end=chunk_end, kwargs={})

        assert result == (chunk_start.value, None)
        assert "Failed chunk" in caplog.text


class TestFetchInChunksParallel:
    """Tests for _fetch_in_chunks_parallel."""

    def test_fetches_and_concatenates_chunks(self):
        """Fetch multiple chunks and concatenate them chronologically."""

        def fetch_fn(start, end):
            index = pd.date_range(start=start, periods=2, freq="h")

            return pd.DataFrame({"load": [100, 200]}, index=index)

        start_date = pd.Timestamp("2026-01-01", tz="Europe/Berlin")
        end_date = pd.Timestamp("2026-01-03", tz="UTC")

        result = _fetch_in_chunks_parallel(
            fetch_fn=fetch_fn,
            start_date=start_date,
            end_date=end_date,
            chunk_days=1,
            max_workers=2,
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert result.index.is_monotonic_increasing
        assert not result.index.duplicated().any()

    def test_removes_duplicate_timestamps(self):
        """Remove duplicated timestamps after concatenation."""
        common_index = pd.date_range(
            "2026-01-01",
            periods=2,
            freq="h",
            tz="UTC",
        )

        calls = []

        def fetch_fn(start, end):
            calls.append((start, end))

            return pd.DataFrame({"load": [100, 200]}, index=common_index)

        start_date = pd.Timestamp("2026-01-01", tz="UTC")
        end_date = pd.Timestamp("2026-01-03", tz="UTC")

        result = _fetch_in_chunks_parallel(
            fetch_fn=fetch_fn,
            start_date=start_date,
            end_date=end_date,
            chunk_days=1,
            max_workers=2,
        )

        assert not result.index.duplicated().any()

    def test_returns_empty_series_when_no_data_is_available(self):
        """Return an empty Series when all chunks fail."""
        fetch_fn = Mock()
        fetch_fn.return_value = pd.DataFrame()

        start_date = pd.Timestamp("2026-01-01", tz="UTC")
        end_date = pd.Timestamp("2026-01-03", tz="UTC")

        result = _fetch_in_chunks_parallel(
            fetch_fn,
            start_date,
            end_date,
            chunk_days=1,
            max_workers=2,
        )

        assert isinstance(result, pd.Series)
        assert result.empty

    def test_naive_timestamps_are_localized_to_utc(self):
        """Localize naive timestamps to UTC."""
        captured = []

        def fetch_fn(start, end):
            captured.append((start, end))
            return pd.DataFrame({"load": [100]}, index=pd.DatetimeIndex([start]))

        start_date = pd.Timestamp("2026-01-01")
        end_date = pd.Timestamp("2026-01-02")

        _fetch_in_chunks_parallel(
            fetch_fn=fetch_fn,
            start_date=start_date,
            end_date=end_date,
            chunk_days=1,
            max_workers=1,
        )

        assert all(str(start.tz) == "UTC" for start, _ in captured)

        assert all(str(end.tz) == "UTC" for _, end in captured)


class TestLoadDataFromEntsoe:
    """Tests for load_data_from_entsoe."""

    @patch("power_quant_strategies.utils.load_data.load_data_from_entsoe.EntsoeClient")
    def test_loads_and_merges_all_datasets(self, mock_client_class):
        """Fetch all datasets and merge them into one DataFrame."""
        index = pd.date_range(
            "2026-01-01",
            periods=3,
            freq="h",
            tz="UTC",
        )

        mock_client = mock_client_class.return_value

        mock_client.query_load_forecast.return_value = pd.DataFrame(
            {"load": [100, 200, 300]},
            index=index,
        )

        mock_client.query_wind_and_solar_forecast.return_value = pd.DataFrame(
            {
                "wind": [100, 200, 300],
                "solar": [5, 10, 15],
            },
            index=index,
        )

        mock_client.query_generation_forecast.return_value = pd.DataFrame(
            {"generation": [120, 230, 340]},
            index=index,
        )

        mock_client.query_day_ahead_prices.return_value = pd.Series(
            [50, 60, 70],
            index=index,
            name="price",
        )

        mock_client.query_imbalance_prices.return_value = pd.Series(
            [45, 55, 65],
            index=index,
            name="imbalance",
        )

        start_date = pd.Timestamp(
            "2026-01-01",
            tz="Europe/Berlin",
        )

        end_date = pd.Timestamp(
            "2026-01-01 02:00",
            tz="Europe/Berlin",
        )

        result = load_data_from_entsoe(
            start_date=start_date,
            end_date=end_date,
            country_code="DE_LU",
        )

        assert isinstance(result, pd.DataFrame)

        assert "load" in result.columns
        assert "wind" in result.columns
        assert "solar" in result.columns
        assert "generation" in result.columns
        assert "da_price" in result.columns
        assert "imb_price" in result.columns

        assert len(result) == 3

        @patch("power_quant_strategies.utils.load_data.load_data_from_entsoe.EntsoeClient")
        def test_uses_de_for_imbalance_price_when_country_is_de_lu(
            self,
            mock_client_class,
        ):
            """Use DE rather than DE_LU for imbalance prices."""
            index = pd.date_range(
                "2026-01-01",
                periods=2,
                freq="h",
                tz="UTC",
            )

            mock_client = mock_client_class.return_value

            mock_client.query_load_forecast.return_value = pd.DataFrame(
                {"load": [100, 200]},
                index=index,
            )

            mock_client.query_wind_and_solar_forecast.return_value = pd.DataFrame({"wind": [10, 20]}, index=index)

            mock_client.query_generation_forecast.return_value = pd.DataFrame({"generation": [100, 200]}, index=index)

            mock_client.query_imbalance_prices.return_value = pd.Series(
                [45, 55],
                index=index,
            )

            load_data_from_entsoe(
                pd.Timestamp("2026-01-01", tz="Europe/Berlin"),
                pd.Timestamp("2026-01-01 01:00", tz="Europe/Berlin"),
                country_code="DE_LU",
            )

            mock_client.query_imbalance_prices.assert_called()
            imbalance_call = mock_client.query_imbalance_prices.call_args
            assert imbalance_call.kwargs["country_code"] == "DE"

            @patch("power_quant_strategies.utils.load_data.load_data_from_entsoe.EntsoeClient")
            def test_returns_empty_dataframe_when_no_forecast_data(
                self,
                mock_client_class,
            ):
                """Return an empty DataFrame when forecast datasets are unavailable."""
                mock_client = mock_client_class.return_value

                mock_client.query_load_forecast.return_value = pd.DataFrame()
                mock_client.query_wind_and_solar_forecast.return_value = pd.DataFrame()
                mock_client.query_generation_forecast.return_value = pd.DataFrame()
                mock_client.query_day_ahead_prices.return_value = pd.Series(dtype="float64")
                mock_client.query_imbalance_prices.return_value = pd.Series(dtype="float64")

                result = load_data_from_entsoe(
                    pd.Timestamp("2026-01-01", tz="Europe/Berlin"),
                    pd.Timestamp("2026-01-02", tz="Europe/Berlin"),
                )

                assert isinstance(result, pd.DataFrame)
                assert result.empty

            @patch("power_quant_strategies.utils.load_data.load_data_from_entsoe.EntsoeClient")
            def test_forward_fills_prices(self, mock_client_class):
                """Forward-fill price data to the forecast index."""
                forecast_index = pd.date_range(
                    "2026-01-01",
                    periods=4,
                    freq="h",
                    tz="UTC",
                )

                price_index = pd.DatetimeIndex(
                    [
                        forecast_index[0],
                        forecast_index[2],
                    ]
                )

                mock_client = mock_client_class.return_value

                mock_client.query_load_forecast.return_value = pd.DataFrame(
                    {"load": [100, 200, 300, 400]},
                    index=forecast_index,
                )

                mock_client.query_wind_and_solar_forecast.return_value = pd.DataFrame(
                    {"wind": [10, 20, 30, 40]},
                    index=forecast_index,
                )

                mock_client.query_generation_forecast.return_value = pd.DataFrame(
                    {"generation": [100, 200, 300, 400]},
                    index=forecast_index,
                )

                mock_client.query_imbalance_prices.return_value = pd.Series(
                    [45, 55],
                    index=price_index,
                )

                result = load_data_from_entsoe(
                    pd.Timestamp("2026-01-01", tz="Europe/Berlin"),
                    pd.Timestamp("2026-01-01 03:00", tz="Europe/Berlin"),
                )

                assert result["da_price"].iloc[0] == 50
                assert result["da_price"].iloc[1] == 50
                assert result["da_price"].iloc[2] == 60
                assert result["da_price"].iloc[3] == 60

                assert result["imb_price"].iloc[0] == 45
                assert result["imb_price"].iloc[1] == 45
                assert result["imb_price"].iloc[2] == 55
                assert result["imb_price"].iloc[3] == 55
