
### This file contains the wrapper class for the LLM model. It recieves a json file, populates metadata(API_key, name, endpoint,etc) and provides methods to interact with the model.
class LLM_Wrapper:
    """
    Json sample:
    {
        "model_name": "gpt2",
        "endpoint": "https://api.openai.com/v1/engines/davinci-codex/completions",
        "API_key": "your-api-key-here",
        "params": {
            "query": "str",
            "max_tokens": "int",
            "temperature": "float",
        },
        
    }
    """
    
    
    def __init__(self, model_metadata: dict)->None:
        pass
        
    def invoke(self, query,context)->str:
        pass
        
    