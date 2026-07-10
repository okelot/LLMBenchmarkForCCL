"""OpenRouter-backed implementation of the LLM_Wrapper interface.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible API, so a single
wrapper can talk to frontier models from every major lab (Anthropic, OpenAI,
Google, xAI, Meta, DeepSeek, ...) by changing only the model id.
"""

import os
import time
from typing import Dict, Optional

from openai import OpenAI
from openai import APIError, BadRequestError

from .llm_wrapper import LLM_Wrapper

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterWrapper(LLM_Wrapper):
    """Non-streaming chat completions via OpenRouter's OpenAI-compatible API."""

    def __init__(self, model_metadata: dict) -> None:
        """
        Args:
            model_metadata: configuration dict containing
                - model_id (str, required): OpenRouter model id, e.g.
                  "anthropic/claude-opus-4.8".
                - api_key (str): OpenRouter key. Falls back to the
                  OPENROUTER_API_KEY environment variable.
                - base_url (str): API base. Defaults to OpenRouter.
                - temperature (float | None): sampling temperature. Defaults to
                  0 for reproducible benchmarks; sent only when not None. If a
                  model rejects it (some reasoning models require the default),
                  the request is retried without it automatically.
                - max_tokens (int | None): completion cap. Omitted when None.
                - timeout (float): per-request timeout in seconds.
                - max_retries (int): automatic retries on transient errors.
                - display_name / developer / context_window /
                  input_price / output_price: optional metadata surfaced by
                  get_model_info().
        """
        super().__init__(model_metadata)

        self.model_id = model_metadata.get("model_id") or model_metadata.get("model_name")
        if not self.model_id:
            raise ValueError("model_metadata must include a 'model_id'.")

        self.api_key = model_metadata.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY or pass 'api_key'."
            )

        self.base_url = model_metadata.get("base_url", DEFAULT_BASE_URL)
        self.temperature = model_metadata.get("temperature", 0)
        self.max_tokens = model_metadata.get("max_tokens")

        # Optional metadata (used only for reporting).
        self.display_name = model_metadata.get("display_name", self.model_id)
        self.developer = model_metadata.get("developer", "")
        self.context_window = model_metadata.get("context_window", "")
        self.input_price = model_metadata.get("input_price", "")
        self.output_price = model_metadata.get("output_price", "")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=model_metadata.get("timeout", 120),
            max_retries=model_metadata.get("max_retries", 4),
            # Optional OpenRouter attribution headers.
            default_headers={
                "HTTP-Referer": "https://github.com/okelot/LLMBenchmarkForCCL",
                "X-Title": "LLM Benchmark for Canadian Case Law",
            },
        )

    def _build_messages(self, query: str, context: Optional[str]):
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": query})
        return messages

    def complete(self, query: str, context: Optional[str] = None) -> Dict:
        """Send a chat completion and return the text plus call metadata.

        Returns a dict: text, prompt_tokens, completion_tokens, finish_reason,
        latency_s. Token counts are 0 when the provider omits usage.
        """
        messages = self._build_messages(query, context)

        kwargs: Dict = {"model": self.model_id, "messages": messages}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        start = time.perf_counter()
        temperature_used = self.temperature
        try:
            if self.temperature is not None:
                try:
                    response = self.client.chat.completions.create(
                        temperature=self.temperature, **kwargs
                    )
                except BadRequestError:
                    # Some models only accept their default temperature. Retry
                    # without it, but record that so results never claim a
                    # decoding config that wasn't actually applied.
                    temperature_used = None
                    response = self.client.chat.completions.create(**kwargs)
            else:
                response = self.client.chat.completions.create(**kwargs)
        except APIError as e:
            raise Exception(f"OpenRouter request failed for {self.model_id}: {e}") from e

        latency = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        text = ""
        finish = None
        if response.choices:
            text = response.choices[0].message.content or ""
            finish = response.choices[0].finish_reason
        return {
            "text": text,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "finish_reason": finish,
            "latency_s": round(latency, 3),
            # decoding/routing transparency
            "temperature_used": "default" if temperature_used is None else temperature_used,
            "served_model": getattr(response, "model", None),
            "provider": getattr(response, "provider", None),
        }

    def invoke(self, query: str, context: Optional[str] = None) -> str:
        """Send a chat completion and return the assistant message content."""
        return self.complete(query, context)["text"]

    def get_model_info(self) -> Dict:
        """Return descriptive metadata about the configured model."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "developer": self.developer,
            "context_window": self.context_window,
            "input_price": self.input_price,
            "output_price": self.output_price,
            "endpoint": self.base_url,
        }
