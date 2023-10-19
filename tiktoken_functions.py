import tiktoken
from typing import List

class Tokenizer:
    def __init__(self, price: float, model_type: str=None, encoding: str="cl100k_base", bundle: int=1000):
        self.model_type = model_type
        if not self.model_type:
            self.encoding = encoding
            self.tokenizer = tiktoken.get_encoding(self.encoding)
        else:
            self.tokenizer = tiktoken.encoding_for_model(self.model_type)
        self.price = price
        self.bundle = bundle
        
        
    def get_cost(self, texts: List[str], return_tokens: bool=False):
        tokens = self.tokenizer.encode_batch(texts)
        if return_tokens:
            return tokens
        else:
            total_tokens = sum([len(chunk) for chunk in tokens])
            cost = (total_tokens/self.bundle) * self.price
            print(f'Total Tokens: {total_tokens:,}\tCost: ${cost:.2f}') 
            return (total_tokens, cost)