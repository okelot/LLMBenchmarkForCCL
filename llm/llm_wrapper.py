"""Abstract base class for LLM provider wrappers.

A wrapper receives a metadata dict (model id, API key, endpoint, sampling
params, ...) and exposes a uniform `invoke(query, context)` method so the
benchmark code stays provider-agnostic. See `openrouter.py` for the concrete
implementation used by this project.
"""

from typing import Dict, Optional


class LLM_Wrapper:
    """
    Metadata sample:
    {
        "model_id": "anthropic/claude-opus-4.8",
        "endpoint": "https://openrouter.ai/api/v1",
        "api_key": "your-api-key-here",
        "temperature": 0,
        "max_tokens": 2048
    }
    """

    def __init__(self, model_metadata: dict) -> None:
        self.model_metadata = model_metadata

    def invoke(self, query: str, context: Optional[str] = None) -> str:
        """Send `query` (with optional system `context`) and return the reply."""
        raise NotImplementedError

    def get_model_info(self) -> Dict:
        """Return descriptive metadata about the configured model."""
        raise NotImplementedError
