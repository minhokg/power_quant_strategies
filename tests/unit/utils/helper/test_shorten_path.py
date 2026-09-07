import os

import pytest

from power_quant_strategies.utils.helper.shorten_path import shorten_path


def test_path_ok_returns_original():
    path = r"C:\tmp\file.png"
    assert shorten_path(path) == path


def test_path_too_long_raises_when_directory_too_long():
    # Construct a path where directory alone exceeds limit
    long_dir = "C:\\" + "a" * 300
    path = os.path.join(long_dir, "file.png")

    with pytest.raises(ValueError):
        shorten_path(path)


def test_trimmed_len_less_than_one():
    # Force very small available filename length
    base_dir = r"C:\tmp"
    # Make directory long enough to almost consume max_path
    long_dir = base_dir + "\\" + ("a" * 240)
    path = os.path.join(long_dir, "verylongfilename.png")

    shortened = shorten_path(path)

    assert shortened.endswith(".png")
    assert len(os.path.basename(shortened).split("_")[0]) == 1


def test_trimmed_len_greater_than_one():
    base_dir = r"C:\tmp"
    long_name = "a" * 300 + ".png"
    path = os.path.join(base_dir, long_name)

    shortened = shorten_path(path)

    assert len(shortened) <= 260
    assert shortened.endswith(".png")
    assert "_" in os.path.basename(shortened)
