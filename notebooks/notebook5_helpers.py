from src.llm.prompt_templates import huberman_system_message, generate_prompt_series
from src.llm.llm_interface import LLM
from typing import Literal
import asyncio


def generate_project2_submission_file(eval_dict: dict, 
                                      outpath: str='./project2_submission.txt',
                                      response_key: str='responses'
                                      ) -> None:
    '''
    Generates a text file for Project 2 submission.  Writes all key-value pairs
    except for the response_key.  Default response_key is 'responses'.
    '''
    with open(outpath, 'w') as f:
        for key, value in eval_dict.items():
            if key != response_key:
                f.write(f'{key}: {value}\n')

async def async_llm_call(llm: LLM,
                          query: str,
                          ranked_result: list[dict],
                          system_message: str=huberman_system_message,
                          verbosity: int=0,
                          temperature: float=1.0,
                          max_tokens: int=250,
                          show_cost: bool=True
                         ) -> tuple[str, float]:
    '''
    Submits async call to LLM.  Creates user message from passed in 
    query and search results. Returns LLM response.
    '''
    user_message = generate_prompt_series(query, ranked_result, verbosity_level=verbosity)
    response = await llm.achat_completion( system_message=system_message,
                                                 user_message=user_message,
                                                 temperature=temperature,
                                                 max_tokens=max_tokens,
                                                 return_cost=show_cost
                                                 )
    return response

async def main(llm: LLM,
               queries: list[str],
               ranked_results: list[list[dict]],
               system_message: str=huberman_system_message,
               verbosity: Literal[0,1,2]=0,
               temperature: float=1.0,
               max_tokens: int=250,
               show_cost: bool=True
              ) -> list[dict]:
    '''
    Submits multiple async LLM calls for execution using asyncio library. 
    By default, prints cost information for multiple calls summed together. 
    '''
    tasks = [async_llm_call(llm, 
                            query, 
                            result, 
                            system_message,
                            verbosity,
                            temperature,
                            max_tokens) 
            for query, result in list(zip(queries, ranked_results))]
    responses = await asyncio.gather(*tasks)
    if show_cost:
        total_cost = sum([r[1] for r in responses])
        responses = [r[0] for r in responses]
        print(f'Total cost for {len(ranked_results)} API calls: ${round(total_cost,4)}.')
    return responses
    
def show_results(queries: list[str], responses: list[str]) -> None:
    '''
    Prints LLM call responses in the following format:
    QUERY: <query>
    RESPONSE: <response>
    --------------------
    '''
    for query, response in zip(queries, responses):
        print(f'QUERY: {query}')
        print(f'RESPONSE: {response}')
        print('-'*100)

def sync_llm_calls(llm: LLM,
                   queries: list[str],
                   ranked_results: list[list[dict]],
                   system_message: str=huberman_system_message,
                   verbosity: int=0,
                   temperature: float=1.0,
                   max_tokens: int=250,
                   show_cost: bool=True
                   ) -> list[str]:
    from tqdm import tqdm
    total_cost = 0
    responses = []
    for i, result in enumerate(tqdm(ranked_results)):

        #create user message on the fly for each result in ranked_results
        user_message = generate_prompt_series(queries[i], result, verbosity_level=verbosity)
        
        #llm call with cost of call returned
        response = llm.chat_completion(system_message=system_message,
                                             user_message=user_message,
                                             temperature=temperature,
                                             max_tokens=max_tokens,
                                             return_cost=show_cost
                                             )
        if show_cost:
            total_cost += response[1]
            response = response[0]
            
        responses.append(response)
    if show_cost:
        print(f'Total cost for {len(ranked_results)} API calls: ${round(total_cost,4)}.')
    return responses