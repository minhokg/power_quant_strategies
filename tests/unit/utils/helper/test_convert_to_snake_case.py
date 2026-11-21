import pytest

from power_quant_strategies.utils.helper.convert_to_snake_case import convert_to_snake_case


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Day Ahead Price", "day_ahead_price"),
        ("Actual Load [MW]", "actual_load_mw"),
        ("  --Wind/Solar--  ", "wind_solar"),
        ("already_snake", "already_snake"),
        ("CamelCase123", "camelcase123"),
        ("", ""),
    ],
)
def test_convert_to_snake_case(value, expected):
    assert convert_to_snake_case(value) == expected
