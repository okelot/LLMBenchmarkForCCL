"""Quick manual smoke test for the OpenRouter wrapper.

    python -m llm.test_openrouter
"""

import os

from dotenv import load_dotenv

from llm.openrouter import OpenRouterWrapper

load_dotenv()

config = {
    "model_id": "anthropic/claude-sonnet-4.6",
    "display_name": "Claude Sonnet 4.6",
    "developer": "Anthropic",
    "api_key": os.environ.get("OPENROUTER_API_KEY"),
    "temperature": 0,
    "max_tokens": 512,
}

wrapper = OpenRouterWrapper(config)
info = wrapper.get_model_info()
print(f"Using {info['display_name']} ({info['model_id']}) via {info['endpoint']}")

response = wrapper.invoke(
    query="In one sentence, what is the ratio decidendi of Donoghue v Stevenson?",
    context="You are a concise legal expert.",
)
print(response)
