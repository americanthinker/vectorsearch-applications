from src.database.weaviate_interface_v4 import WeaviateWCS
import os

def get_weaviate_client(endpoint: str=os.environ.get('WEAVIATE_ENDPOINT'),
                        api_key: str=os.environ.get('WEAVIATE_API_KEY'),
                        model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2',
                        embedded: bool=False,
                        openai_api_key: str=None,
                        **kwargs
                        ) -> WeaviateWCS:
    return WeaviateWCS(endpoint, api_key, model_name_or_path, embedded, openai_api_key, **kwargs)