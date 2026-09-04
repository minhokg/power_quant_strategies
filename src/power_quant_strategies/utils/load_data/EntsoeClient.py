import os

from entsoe import EntsoePandasClient
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class EntsoeClient(EntsoePandasClient):
    """Entsoe Client with expanded connection pools and robust exponential retries."""

    def __init__(self) -> None:
        """
        Initialize the ENTSO-E client with API authentication and retry settings.

        The API key is loaded from the ``ENTSOE_API_KEY`` environment variable.
        The underlying ENTSO-E client is configured with a 300-second timeout,
        an expanded HTTP connection pool, and an exponential backoff retry
        strategy for transient API and server errors.

        :raises EnvironmentError:
            If the ``ENTSOE_API_KEY`` environment variable is not set.
        """
        api_key = os.getenv("ENTSOE_API_KEY")

        if api_key is None:
            raise EnvironmentError("You have to set the environment variable ENTSOE_API_KEY.")

        super().__init__(api_key=api_key, timeout=300)

        # Configure exponential backoff retry strategy for transient ENTSO-E errors
        retry_strategy = Retry(
            total=5,  # Retry up to 5 times
            backoff_factor=2,  # Wait 2s, 4s, 8s, 16s... between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry
            raise_on_status=False,
        )

        # Mount the retry adapter with expanded pool size
        adapter = HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=retry_strategy)

        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
