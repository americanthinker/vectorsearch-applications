import tiktoken
from tiktoken import Encoding

def get_token_count(content: str | list[str], 
                    encoder: Encoding=None, 
                    encoding: str='cl100k_base', 
                    return_tokens: bool=False
                    ) -> int | list[int]:
    '''
    Takes a tiktoken encoder and returns token count for a given message(s).
    If return_tokens is True, it will return a list of tokens and print the count.
    '''
    if encoder == None:
        encoder = tiktoken.get_encoding(encoding)
    if isinstance(content, str):
        tokens = encoder.encode(content)
        count = len(tokens)
    elif isinstance(content, list):
        tokens = encoder.encode_batch(content)
        count = sum(list(map(len, tokens)))
    if return_tokens:
        print(f'Total tokens: {count}')
        return tokens
    return count