import os
from llm_lod import ClodWrapper
# Configuration
config = {
    "model_name": "gpt-4o",
    "endpoint": "https://api.clod.io/v1",
    "API_key": os.environ.get('CLOD_API_KEY'),
    "models_csv_path": "ai_models_lod.csv"  # Path to the CSV we created earlier
}

# Create wrapper instance
wrapper = ClodWrapper(config)

# Get model information
model_info = wrapper.get_model_info()
print(f"Using {model_info['model_name']} from {model_info['vendor']}")
print(f"Price per million tokens: ${model_info['token_in_price']} in, ${model_info['token_out_price']} out")

# Make a request
response = wrapper.invoke(
    query="What is quantum computing?",
    context="Provide a simple explanation suitable for beginners"
)
print(response)