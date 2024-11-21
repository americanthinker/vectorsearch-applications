import tiktoken
from tiktoken import Encoding
from src.llm.llm_interface import LLM
import os

def get_token_count(content: str | list[str], 
                    encoder: Encoding=None, 
                    encoding: str='o200k_base', 
                    return_tokens: bool=False,
                    verbose: bool=True
                    ) -> int | list[int]:
    '''
    Takes a tiktoken encoder and returns token count for a given message(s).
    If return_tokens is True, it will return a list of tokens and print the count.
    '''
    if encoder is None:
        encoder = tiktoken.get_encoding(encoding)
    if isinstance(content, str):
        tokens = encoder.encode(content)
        count = len(tokens)
    elif isinstance(content, list):
        tokens = encoder.encode_batch(content)
        count = sum(list(map(len, tokens)))
    if return_tokens:
        if verbose:
            print(f'Total tokens: {count}')
        return tokens
    return count

def load_azure_openai(model_name: str='gpt-35-turbo', 
                      api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                      api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                      api_base=os.getenv('AZURE_OPENAI_ENDPOINT')
                      ) -> LLM:
    '''
    Loads an Azure OpenAI LLM from preset defaults.
    '''
    llm = LLM(model_name=f'azure/{model_name}', 
              api_key=api_key, 
              api_version=api_version, 
              api_base=api_base)
    return llm