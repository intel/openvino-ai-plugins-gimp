from json import loads
from http import client
from urllib.parse import urlparse


class FastSDApiClient:
    def __init__(
        self,
        server_url: str,
    ):
        self.server_url = server_url
        self.url = urlparse(self.server_url)

    def _get_request(
        self,
        url: str,
    ) -> dict:
        try:
            conn = client.HTTPConnection(self.url.hostname, self.url.port)
            headers = {"Content-Type": "application/json"}
            conn.request(
                "GET",
                url,
                body=None,
                headers=headers,
            )
            res = conn.getresponse()
            data = res.read()
            result = loads(data)
            return result
        except Exception as exception:
            raise RuntimeError(f"Error: {str(exception)}")

    def load_settings(self) -> dict:
        """Loads FastSD settings"""
        try:
            config = self._get_request("/api/config")
            return config
        except Exception as exception:
            raise RuntimeError(
                f"Failed to get settings! {str(exception)}"
            ) from exception

    def get_info(self) -> dict:
        """
        Returns the system info of the FastSD server.
        """
        try:
            result = self._get_request("/api/info")
            return result
        except Exception as exception:
            raise RuntimeError(
                f"Failed to get info from FastSD! {str(exception)}"
            ) from exception

    def get_models(self) -> list:
        """
        Returns a list of available models.
        """
        try:
            result = self._get_request("/api/models")
            return result["openvino_models"]
        except Exception as exception:
            raise RuntimeError(
                f"Failed to get models from API! {str(exception)}"
            ) from exception

    def generate_text_to_image(
        self,
        config: dict,
    ) -> dict:
        """Generates an image based on the provided configuration."""
        try:
            conn = client.HTTPConnection(self.url.hostname, self.url.port)
            headers = {"Content-Type": "application/json"}
            conn.request("POST", "/api/generate", config, headers)
            res = conn.getresponse()
            data = res.read()
            result = loads(data)
            return result
        except Exception as exception:
            raise RuntimeError(
                f"Failed to generate image! {str(exception)}"
            ) from exception
