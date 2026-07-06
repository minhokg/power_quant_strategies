import logging

import psutil


def get_usable_cpu_count(
    percentage: float = 1.0,
) -> int:
    """
    Get usable cpu count.

    We use only # of physical cpus because hyperthreading core often do not provide performance benefits.

    :param percentage: percentage of non-used cpus to use, default is 1.0 (all non-used cpus)
    :return: usable cpu count, defaults to all available physical cpus
    """
    logging.info("get usable physical cpu count")
    cpu_count = psutil.cpu_count(logical=False)
    if cpu_count is None:
        logging.warning("Could not determine physical cpu count, fallback to logical cpu count")
        return 1
    return max(1, round(cpu_count * percentage))
