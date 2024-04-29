from dotenv import load_dotenv, find_dotenv
envs = load_dotenv(find_dotenv(), override=True)

import sys
import os
sys.path.append('../')

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset


from src.database.database_utils import get_weaviate_client
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.llm.llm_interface import LLM
from src.llm.prompt_templates import huberman_system_message, generate_prompt_series

from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from litellm import ModelResponse
from math import ceil
import asyncio
from rich import print

#set constants
questions = ["Give a brief explanation of how brain neuroplasticity works",
             "What is the role of dopamine in the body",
             "What is a catecholimine",
             "What does Jocko have to say about leadership",
             "What does Fridman think about the evolution of AI", 
             "Who is the host of the Huberman Labs podcast",
             "Why do people make self-destructive decisions",
             "Provide better sleep protocol in list format",
             "What are the topcis that Lex Fridman discusses",
             "Is there a generally positive outlook on the future of AI",
            ]

logger.info('Setting constants...')
client = get_weaviate_client()
turbo = LLM(model_name='gpt-3.5-turbo-0125')
collection_name = 'Huberman_minilm_128'
arm, fm = AnswerRelevancyMetric(model='gpt-4', threshold=0.7), FaithfulnessMetric(model='gpt-4', threshold=0.7)

def get_answer_bundle(query: str,
                      client: WeaviateWCS,
                      collection_name: str,
                      answer_llm: LLM,
                      ground_truth_llm: LLM=None
                     ) -> tuple[str, list[list[str]], str]:
    '''
    Returns answer, ground truth and associated context from a single query.
    '''
    def format_llm_response(response: ModelResponse) -> str:
        return response.choices[0].message.content

    #1st-stage retrieval (get contexts)
    context = client.hybrid_search(query, collection_name, 
                                   query_properties=['content', 'title', 'short_description'],
                                   limit=3, 
                                   return_properties=['content', 'guest', 'short_description'])
    #create contexts from content field
    contexts = [d['content'] for d in context]
    
    #generate assistant message prompt
    assist_message = generate_prompt_series(query, context, summary_key='short_description')

    #generate answers from model being evaluated
    answer = format_llm_response(answer_llm.chat_completion(huberman_system_message, assist_message))

    #create ground truth answers
    if ground_truth_llm:
        ground_truth = format_llm_response(ground_truth_llm.chat_completion(huberman_system_message, assist_message))
        return query, contexts, answer, ground_truth
    return query, contexts, answer


async def create_test_dataset(questions: list[str], 
                              client: WeaviateWCS,
                              collection_name: str,
                              answer_llm: LLM,
                              ground_truth_llm: LLM=None, 
                              batch_size: int=5, 
                              async_mode: bool=True,
                              disable_internal_tqdm: bool=False):
    total = len(questions)
    progress = tqdm('Queries', total=total, disable=disable_internal_tqdm)
    data = []
    batches = ceil(total/batch_size)
    for i in range(batches):
        batch = questions[i*batch_size:(i+1)*batch_size]
        if async_mode:
            results = await asyncio.gather(*[aget_answer_bundle(query, 
                                                                client, 
                                                                collection_name, 
                                                                answer_llm,
                                                                ground_truth_llm) for query in batch])
            if any(results):
                data.extend(results)
            else:
                raise "No results returned for initial batch, double-check your inputs."
        else:
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                futures = [executor.submit(get_answer_bundle, query, client, collection_name, answer_llm, ground_truth_llm) for query in batch]
                for future in as_completed(futures):
                    progress.update(1)
                    data.append(future.result())
        print(f"Finished with batch {i+1}, taking a break...")
    queries = [d[0] for d in data]
    contexts = [d[1] for d in data]
    answers = [d[2] for d in data]
    if len(data[0]) == 4:
        ground_truths = [d[3] for d in data]
        return queries, contexts, answers, ground_truths
    return queries, contexts, answers

async def aget_answer_bundle( query: str,
                              client: WeaviateWCS,
                              collection_name: str,
                              answer_llm: LLM,
                              ground_truth_llm: LLM=None
                             ) -> tuple[str, list[list[str]], str]:
    '''
    Returns answer, ground truth and associated context from a single query.
    '''
    def format_llm_response(response: ModelResponse) -> str:
        return response.choices[0].message.content

    #1st-stage retrieval (get contexts)
    context = client.hybrid_search(query, collection_name, 
                                   query_properties=['content', 'title', 'short_description'],
                                   limit=3, 
                                   return_properties=['content', 'guest', 'short_description'])
    #create contexts from content field
    contexts = [d['content'] for d in context]
    
    #generate assistant message prompt
    assist_message = generate_prompt_series(query, context, summary_key='short_description')

    #generate answers from model being evaluated
    answer = await answer_llm.achat_completion(huberman_system_prompt, assist_message)
    answer = format_llm_response(answer)

    #create ground truth answers
    if ground_truth_llm:
        ground_truth = format_llm_response(ground_truth_llm.chat_completion(huberman_system_prompt, assist_message))
        return query, contexts, answer, ground_truth
    return query, contexts, answer

logger.info('Creating dataset...')
data = asyncio.run(create_test_dataset(questions[:5], client, collection_name, turbo))

queries, contexts, answers = data[0], data[1], data[2]
logger.info(queries)

def create_eval_dataset(questions: list[str],
                        contexts: list[list[str]],
                        answers: list[str]
                       ) -> EvaluationDataset:
    assert len(questions) == len(contexts) == len(answers), 'Mismatched lengths in input values, retry after correcting'
    test_cases = []
    for i in range(len(questions)):
        test_case = LLMTestCase(input=questions[i],
                                actual_output=answers[i],
                                retrieval_context=contexts[i])
        test_cases.append(test_case)
    return EvaluationDataset(alias='Initial test', test_cases=test_cases)

logger.info("Creating evaluation dataset...")
eval_dataset = create_eval_dataset(queries, contexts, answers)

evaluation = evaluate(eval_dataset, metrics=[fm], print_results=False)
for line in evaluation:
    logger.info(vars(line.metrics[0]))

import json
try:
    logger.info('Saving as json...')
    with open('temp_results.json', 'w') as f:
        f.write(json.dumps(evaluation))
        
except Exception:
    logger.info('Saving as text...')
    with open('temp_results.txt', 'w') as f:
        for line in evaluation:
            _metrics = line.metrics[0].__dict__
            f.write(f'{_metrics}\n')




