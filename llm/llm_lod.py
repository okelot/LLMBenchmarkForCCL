from .llm_wrapper import LLM_Wrapper
import requests
import pandas as pd
from typing import Optional, Dict
from pathlib import Path

class ClodWrapper(LLM_Wrapper):
    """
    Wrapper for the clod.io API service.
    Reads model configurations from CSV and provides non-streaming completions.
    """
    
    def __init__(self, model_metadata: dict) -> None:
        """
        Initialize the wrapper with model metadata.
        
        Args:
            model_metadata (dict): Configuration including:
                - model_name: Name of the model to use
                - endpoint: Base API endpoint
                - API_key: Authentication key
                - models_csv_path: Path to the CSV file containing model information
        """
        super().__init__(model_metadata)
        
        # Store configuration
        self.model_name = model_metadata.get('model_name')
        self.base_endpoint = model_metadata.get('endpoint', 'https://api.clod.io/v1')
        self.api_key = model_metadata.get('API_key')
        self.models_csv_path = model_metadata.get('models_csv_path')
        
        # Load models from CSV
        self.models_info = self._load_models_from_csv()
        
        # Set up headers
        self.headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        # Validate model configuration
        if self.model_name not in self.models_info:
            raise ValueError(f"Unsupported model: {self.model_name}. Available models: {list(self.models_info.keys())}")

    def _load_models_from_csv(self) -> Dict:
        """
        Load and parse the models information from CSV.
        
        Returns:
            Dict: Mapping of model names to their configurations
        """
        if not Path(self.models_csv_path).exists():
            raise FileNotFoundError(f"Models CSV file not found: {self.models_csv_path}")
            
        df = pd.read_csv(self.models_csv_path)
        
        models_info = {}
        for _, row in df.iterrows():
            models_info[row['Model Name']] = {
                'vendor': row['Provider Name'],
                'model': row['Name in Provider'],
                'context_window': row['Context Window'],
                'token_in_price': row['Token IN Price ($)'],
                'token_out_price': row['Token OUT Price ($)']
            }
            
        return models_info

    def _create_chat_request(self, query: str, context: Optional[str] = None) -> Dict:
        """
        Create the chat request payload.
        
        Args:
            query (str): The user's query
            context (str, optional): Additional context for the query
            
        Returns:
            Dict: The formatted request payload
        """
        messages = []
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": context
            })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Get model information
        model_info = self.models_info[self.model_name]
        
        return {
            "vendor": model_info['vendor'],
            "model": model_info['model'],
            "ensure_success": True,
            "messages": messages,
            "stream": False
        }

    def invoke(self, query: str, context: Optional[str] = None) -> str:
        """
        Send a completion request to the API.
        
        Args:
            query (str): The user's query
            context (str, optional): System context or instructions
            
        Returns:
            str: The model's response
        """
        endpoint = f"{self.base_endpoint}/chat/completions"
        payload = self._create_chat_request(query, context)
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def get_model_info(self) -> Dict:
        """
        Get information about the currently configured model.
        
        Returns:
            Dict: Model configuration information including pricing and context window
        """
        model_info = self.models_info[self.model_name]
        return {
            "model_name": self.model_name,
            "vendor": model_info['vendor'],
            "model_id": model_info['model'],
            "context_window": model_info['context_window'],
            "token_in_price": model_info['token_in_price'],
            "token_out_price": model_info['token_out_price'],
            "endpoint": self.base_endpoint
        }