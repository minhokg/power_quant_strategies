import pytest

from power_quant_strategies.utils.helper import get_usable_cpu_count as module


@pytest.mark.parametrize(("physical", "percentage", "expected"), [(8, 1.0, 8), (8, 0.5, 4), (8, 0.01, 1), (3, 0.5, 2)])
def test_scales_physical_cpus(monkeypatch, physical, percentage, expected):
    monkeypatch.setattr(module.psutil, "cpu_count", lambda logical: physical)
    assert module.get_usable_cpu_count(percentage=percentage) == expected


def test_falls_back_to_one_when_unknown(monkeypatch):
    monkeypatch.setattr(module.psutil, "cpu_count", lambda logical: None)
    assert module.get_usable_cpu_count() == 1


def test_asks_for_physical_cpus(monkeypatch):
    calls = []

    def cpu_count(logical):
        calls.append(logical)
        return 4

    monkeypatch.setattr(module.psutil, "cpu_count", cpu_count)
    module.get_usable_cpu_count()
    assert calls == [False]
