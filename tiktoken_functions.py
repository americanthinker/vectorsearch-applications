import tiktoken
from typing import List

class Tokenizer:
    def __init__(self, price: float, model_type: str="cl100k_base", bundle: int=1000):
        self.model_type = model_type
        self.price = price
        self.bundle = bundle
        self.tokenizer = tiktoken.get_encoding(model_type)
        
    def get_cost(self, texts: List[str], return_tokens: bool=False):
        tokens = self.tokenizer.encode_batch(texts)
        if return_tokens:
            return tokens
        else:
            total_tokens = sum([len(chunk) for chunk in tokens])
            cost = (total_tokens/self.bundle) * self.price
            print(f'Total Tokens: {total_tokens}\tCost: {cost:.3f}') 
            return (total_tokens, cost)