import time
from typing import List
import tiktoken 
from loguru import logger
from prompt_templates import context_block, question_answering_prompt_series

def convert_seconds(seconds: int):
    """
    Converts seconds to a string of format Hours:Minutes:Seconds
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def generate_prompt(base_prompt: str, query: str, results: List[dict]) -> str:
    """
    Generates a prompt for the OpenAI API
    """
    contexts = '\n\n'.join([r['content'] for r in results])
    prompt = base_prompt.format(question=query, context=contexts)
    return prompt

def generate_prompt_series(query: str, results: List[dict]) -> str:
    """
    Generates a prompt for the OpenAI API
    """
    context_series = '\n'.join([context_block.format(summary=res['summary'], guest=res['guest'], \
                                         transcript=res['content']) for res in results]).strip()
    prompt = question_answering_prompt_series.format(question=query, series=context_series)
    return prompt

def validate_token_threshold(ranked_results: List[dict], 
                             base_prompt: str,
                             query: str,
                             tokenizer: tiktoken.Encoding, 
                             token_threshold: int, 
                             verbose: bool = False
                             ) -> List[dict]:
        """
        Validates that prompt is below the set token threshold by adding lengths of:
            1. Base prompt
            2. User query
            3. Context material
        If threshold is exceeded, context results are reduced incrementally until the 
        combined prompt tokens are below the threshold. 
        """
        overhead_len = len(tokenizer.encode(base_prompt.format(question=query, series='')))
        context_len = get_batch_length(ranked_results, tokenizer)
    
        token_count = overhead_len + context_len
        if token_count > token_threshold:
            print('Token count exceeds token count threshold, reducing size of returned results below token threshold')
            
            while token_count > token_threshold and len(ranked_results) > 1:
                num_results = len(ranked_results)
                
                # remove the last ranked (most irrelevant) result
                ranked_results = ranked_results[:num_results-1]
                # recalculate new token_count
                token_count = overhead_len + get_batch_length(ranked_results, tokenizer)

        if verbose:
            logger.info(f'Total Final Token Count: {token_count}')
        return ranked_results

def get_batch_length(ranked_results: List[dict], tokenizer: tiktoken.Encoding) -> int:
    contexts = tokenizer.encode_batch([r['content'] for r in ranked_results])
    context_len = sum(list(map(len, contexts)))
    return context_len