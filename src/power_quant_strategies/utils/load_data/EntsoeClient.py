import os

from entsoe import EntsoePandasClient


class EntsoeClient(EntsoePandasClient):
    """Initialize the Entsoe Client."""

    def __init__(self) -> None:
        """Initialize the Entsoe Client."""
        # get api key to authorize
        api_key = os.getenv("ENTSOE_API_KEY")

        # make sure api key is available to connect
        if api_key is None:
            raise EnvironmentError("You have to set the environment variable ENTSOE_API_KEY.")

        # initialize with supper class
        super().__init__(api_key=api_key, timeout=300)
