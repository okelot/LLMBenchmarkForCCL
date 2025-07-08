import requests
from typing import Dict, List
import csv
from datetime import datetime

BASE_URL = "https://api.clod.io/v1/providers/models"
import os

API_KEY = os.environ.get('CLOD_API_KEY')
if not API_KEY:
    raise ValueError('CLOD_API_KEY environment variable is not set')

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def get_models() -> List[Dict]:
    """Fetch all models from the API."""
    response = requests.get(BASE_URL, headers=headers)
    response.raise_for_status()
    return response.json()

def write_to_csv(models_data: List[Dict]):
    """Write the models data to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_models_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'Provider Name',
            'Provider Website',
            'Provider Base URL',
            'Model Name',
            'Name in Provider',
            'Developer',
            'System Complexity',
            'Context Window',
            'Token IN Price ($)',
            'Token OUT Price ($)',
            'Active',
            'Created At',
            'Updated At'
        ])
        
        # Write data
        for model in models_data:
            provider = model.get('provider', {})
            writer.writerow([
                provider.get('name', ''),
                provider.get('websiteUrl', ''),
                provider.get('baseUrl', ''),
                model.get('systemName', ''),
                model.get('nameInProvider', ''),
                model.get('developer', ''),
                model.get('systemComplexity', ''),
                model.get('contextWindow', ''),
                float(model.get('tokenInPricePerMillionTokens', 0)),
                float(model.get('tokenOutPricePerMilionTokens', 0)),
                'Yes' if model.get('active') else 'No',
                model.get('createdAt', ''),
                model.get('updatedAt', '')
            ])
    
    print(f"Data written to {filename}")
    return filename

def analyze_models(models_data: List[Dict]):
    """Analyze the models data and print summary statistics."""
    providers = {}
    developers = {}
    total_models = len(models_data)
    
    for model in models_data:
        # Count models per provider
        provider_name = model['provider']['name']
        providers[provider_name] = providers.get(provider_name, 0) + 1
        
        # Count models per developer
        developer = model.get('developer', 'Unknown')
        developers[developer] = developers.get(developer, 0) + 1
    
    print("\nSummary Statistics:")
    print(f"Total number of models: {total_models}")
    
    print("\nModels by Provider:")
    for provider, count in sorted(providers.items()):
        print(f"{provider}: {count} models")
        
    print("\nModels by Developer:")
    for developer, count in sorted(developers.items()):
        print(f"{developer}: {count} models")

def main():
    try:
        print("Fetching models data...")
        models_data = get_models()
        print(f"Found {len(models_data)} models.")
        
        # Write to CSV
        filename = write_to_csv(models_data)
        
        # Print analysis
        analyze_models(models_data)
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()