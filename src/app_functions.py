import time
import json
from typing import Literal, Generator, Any
import tiktoken 
from time import sleep
from loguru import logger
from src.llm.llm_interface import LLM
from src.llm.prompt_templates import (context_block, question_answering_prompt_series, 
                                      generate_prompt_series, verbosity_options, huberman_system_message)
import streamlit as st  

@st.cache_data
def load_data(data_path: str) -> dict | list[dict]:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def convert_seconds(seconds: int) -> str:
    """
    Converts seconds to a string of format Hours:Minutes:Seconds
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def validate_token_threshold(ranked_results: list[dict],
                             query: str,
                             system_message: str,
                             tokenizer: tiktoken.Encoding, 
                             token_threshold: int,
                             llm_verbosity_level: Literal[0, 1, 2]=0,
                             content_field: str='content', 
                             verbose: bool = False
                             ) -> list[dict]:
        """
        Validates that prompt is below the set token threshold by adding lengths of:
            1. Base prompt
            2. User query
            3. Context material
        If threshold is exceeded, context results are reduced incrementally until the 
        combined prompt tokens are below the threshold. This function does not take into
        account every token passed to the LLM, but it is a good approximation.
        """
        user_message = generate_prompt_series(query=query, results=ranked_results, verbosity_level=llm_verbosity_level)
        user_len = len(tokenizer.encode(user_message))
        system_len = len(tokenizer.encode(system_message))
        # context_len = _get_batch_length(ranked_results, tokenizer, content_field=content_field)
    
        token_count = user_len + system_len
        if token_count > token_threshold:
            logger.info('Token count exceeds token count threshold, reducing size of returned results below token threshold')
            
            while token_count > token_threshold and len(ranked_results) > 1:
                num_results = len(ranked_results)
                
                # remove the last ranked (most irrelevant) result
                ranked_results = ranked_results[:num_results-1]
                # recalculate new token_count
                user_message = generate_prompt_series(query=query, results=ranked_results, verbosity_level=llm_verbosity_level)
                token_count = len(tokenizer.encode(user_message)) + system_len

        if verbose:
            logger.info(f'Total Final Token Count: {token_count}')
        return ranked_results

def _get_batch_length(ranked_results: list[dict], 
                      tokenizer: tiktoken.Encoding, 
                      content_field: str='content'
                      ) -> int:
    '''
    Convenience function to get the length in tokens of a batch of results 
    '''
    contexts = tokenizer.encode_batch([r[content_field] for r in ranked_results])
    context_len = sum(list(map(len, contexts)))
    return context_len

def stream_chat(
                llm: LLM,
                user_message: str,
                system_message: str=huberman_system_message,
                max_tokens: int=250,
                temperature: float=0.5,
                ) -> Generator[Any, Any, None]:
    """Generate chat responses using an LLM API.
    Stream response out to UI.
    """
    completion = llm.chat_completion(system_message=system_message,
                                     user_message=user_message,
                                     temperature=temperature, 
                                     max_tokens=max_tokens, 
                                     stream=True)
    for chunk in completion:
        sleep(0.05)
        if any(chunk.choices):
            content = chunk.choices[0].delta.content
            if content:
                yield content
    return completion

def stream_json_chat(llm: LLM,
                     user_message: str,
                     system_message: str=huberman_system_message,
                     max_tokens: int=250,
                     temperature: float=0.5
                     ) -> Generator[Any, Any, None]:
    """Generate chat responses using an LLM API.
    Stream response out to UI.
    """

    completion = llm.chat_completion(system_message=system_message,
                                     user_message=user_message,
                                     temperature=temperature, 
                                     max_tokens=max_tokens, 
                                     stream=True,
                                     response_format={ "type": "json_object" })
    colon_count = 0
    full_json = []
    double_quote_count = 0
    double_quote = '"'
    colon = ":"

    for chunk in completion:
        sleep(0.05)
        if any(chunk.choices):
            content = chunk.choices[0].delta.content
            if content:
                full_json.append(content)
                if colon in content:
                    colon_count += 1
                    continue
            if colon_count >= 1:
                if double_quote_count == 2:
                    continue
                if double_quote_count < 2:
                    yield content
                if content and double_quote in content:
                    double_quote_count += 1

    try:
        answer = json.loads("".join(full_json))["answer"]
        guest = json.loads("".join(full_json))["guest"]
        logger.info(f'GUEST: {guest}')
        logger.info(f'ANSWER: {answer}')
    except Exception as e:
        json_keys = json.loads("".join(full_json)).keys()
        logger.info(f'Exception raised in JSON parsing: {e}. Keys were parsed as: {json_keys}')
    json_response = json.loads("".join(full_json))
    return json_response

def search_result(i: int, 
                  url: str, 
                  title: str, 
                  content: str,
                  guest: str,
                  length: str,
                  space: str='&nbsp; &nbsp;'
                 ) -> str:
    
    '''
    HTML to display search results.

    Args:
    -----
    i: int
        index of search result
    url: str
        url of YouTube video 
    title: str
        title of episode 
    content: str
        content chunk of episode
    '''
    return f"""
        <div style="font-size:120%;">
            {i + 1}.<a href="{url}">{title}</a>
        </div>

        <div style="font-size:95%;">
            <p>Episode Length: {length} {space}{space} Guest: {guest}</p>
            <div style="color:grey;float:left;">
                ...
            </div>
            {content}
        </div>
    """