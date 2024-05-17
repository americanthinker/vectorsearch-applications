from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import GEval, BaseMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.evaluate import TestResult
from deepeval import evaluate

from anthropic import Anthropic, AsyncAnthropic
from cohere import Client, AsyncClient
from openai import AzureOpenAI, AsyncAzureOpenAI

from src.reranker import ReRanker
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.llm.llm_interface import LLM
from src.llm.prompt_templates import huberman_system_message, generate_prompt_series, create_context_blocks
from tqdm.asyncio import tqdm_asyncio 
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal, Any, Optional
from math import ceil
import numpy as np
import os

class CustomCohere(DeepEvalBaseLLM):
    """
    Creates a custom evaluation model interface that uses the Cohere API
    for evaluation of metrics. 
    """

    def __init__(
                self,
                model: Literal['command-r', 'command-r-plus']
                ) -> None:
        self.accepted_model_types = ['command-r', 'command-r-plus']
        if model not in self.accepted_model_types:
            raise ValueError(f'{model} is not found in {self.accepted_model_types}. Retry with acceptable model type')
        self.model = model

    def load_model(self, async_mode: bool=False) -> Client | AsyncClient:
        if async_mode:
            return AsyncClient(api_key=os.getenv('COHERE_API_KEY'))
        return Client(api_key=os.getenv('COHERE_API_KEY'))

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        response = client.chat(message=prompt, model=self.model, max_tokens=1024)
        if response:
            return response.text
        return "No message returned"

    async def a_generate(self, prompt: str) -> str:
        aclient = self.load_model(async_mode=True)
        response = await aclient.chat(message=prompt, model=self.model, max_tokens=1024)
        if response:
            return response.text
        return "No message returned"

    def get_model_name(self) -> str:
        return self.model
    
class CustomAnthropic(DeepEvalBaseLLM):
    """
    Creates a custom evaluation model interface that uses the Anthropic API 
    for evaluation of metrics. 
    """
    def __init__(
                self,
                model: Literal['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229']
                ):
        self.accepted_model_types = ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229']
        if model not in self.accepted_model_types:
            raise ValueError(f'{model} is not found in {self.accepted_model_types}. Retry with acceptable model type')
        self.model = model

    def load_model(self, async_mode: bool=False) -> AsyncAnthropic | Anthropic:
        if async_mode:
            return AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        return Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        message = client.messages.create(
                                        max_tokens=1024,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": prompt,
                                            }
                                        ],
                                        model=self.model,
                                        )
        if message:
            return message.content[0].text
        return "no message returned"

    async def a_generate(self, prompt: str) -> str:
        aclient = self.load_model(async_mode=True)
        message = await aclient.messages.create(
                                                max_tokens=1024,
                                                messages=[
                                                    {
                                                        "role": "user",
                                                        "content": prompt,
                                                    }
                                                ],
                                                model=self.model,
                                                )
        if message:
            return message.content[0].text
        return "no message returned"

    def get_model_name(self) -> str:
        return self.model
    

class CustomAzureOpenAI(DeepEvalBaseLLM):
    def __init__(
                self,
                deployment_name: str
                ) -> None:
        self.model = deployment_name

    def load_model(self, async_mode: bool=False) -> AzureOpenAI | AsyncAzureOpenAI:
        if async_mode:
            return AsyncAzureOpenAI(azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                                    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                                    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                                    )
        return AzureOpenAI(azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                           api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                           api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                           )

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        completion = client.chat.completions.create(model=self.model,
                                                    messages=[
                                                            {
                                                        "role": "user",
                                                        "content": prompt,
                                                            }
                                                        ],
                                                    max_tokens=1024)
        if completion:
            return completion.choices[0].message.content
        return "no message returned"

    async def a_generate(self, prompt: str) -> str:
        aclient = self.load_model(async_mode=True)
        completion = await aclient.chat.completions.create(model=self.model,
                                                            messages=[
                                                                {
                                                                    "role": "user",
                                                                    "content": prompt,
                                                                }
                                                            ],
                                                            max_tokens=1024)
        if completion:
            return completion.choices[0].message.content
        return "no message returned"

    def get_model_name(self) -> str:
        return self.model
    

class AnswerCorrectnessMetric(GEval):
    '''
    Custom metric to determine correctness of an LLM generated answer
    as related to the retrieval context. Takes in LLM model string as 
    single param. 
    '''
    name = 'AnswerCorrectness'
    evaluation_steps=[  'Compare the actual output with the retrieval context to verify factual accuracy.',
                        'Assess if the actual output effectively addresses the specific information requirement stated in the input.',
                        'Determine the comprehensiveness of the actual output in addressing all key aspects mentioned in the input.',
                        # 'Score the actual output between 0 and 10, based on the accuracy and comprehensiveness of the information provided, with 10 being exactly correct and 0 being completely incorrect.',
                        'If there is not enough information in the retrieval context to correctly answer the input, and the actual output indicates that the input cannot be answered with the provided context, then return a score of 10.']
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]

    def __init__(self, model: str | DeepEvalBaseLLM) -> None:
        self.model = model
        super().__init__(name=self.name,
                         evaluation_steps=self.evaluation_steps,
                         model=self.model,
                         evaluation_params=self.evaluation_params)

@dataclass
class EvalResponse:
    score: float
    reason: str
    metric: str
    cost: float
    eval_model: str
    eval_steps: Optional[list[str]] = None
    verdicts: Optional[list[str]] = None    
    input: Optional[str] = None
    actual_output: Optional[str] = None
    retrieval_context: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return self.__dict__


def load_eval_response(metric: BaseMetric | AnswerCorrectnessMetric,
                       test_case: LLMTestCase | TestResult,
                       return_context_data: bool=True
                       ) -> EvalResponse:
    '''
    Parses and loads select data from metric and test_case and 
    combines into a single EvalResponse package for ease of viewing. 
    '''
    if isinstance(test_case, TestResult) and isinstance(metric, list):
        if len(metric) > 1:
            raise NotImplementedError("Multiple metrics not supported yet for this loading function")
        metric = metric[0]
    return EvalResponse(score=metric.score,
                        reason=metric.reason,
                        metric=metric.__class__.__name__,
                        cost=metric.evaluation_cost, 
                        eval_model=metric.evaluation_model,
                        eval_steps=metric.evaluation_steps if metric.__dict__.get('evaluation_steps') else None,
                        verdicts=metric.verdicts if metric.__dict__.get('verdicts') else None,
                        input=test_case.input if return_context_data else None,
                        actual_output=test_case.actual_output if return_context_data else None,
                        retrieval_context=test_case.retrieval_context if return_context_data else None
                        )

class TestCaseGenerator:

    def __init__(self, 
                 llm: LLM,
                 retriever: WeaviateWCS,
                 reranker: ReRanker
                 ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker

    def retrieve_results(self,
                         queries: list[str],
                         collection_name: str,
                         limit: int=200,
                         top_k: int=3
                         ) -> list[dict]:
        results = [self.retriever.hybrid_search(query, collection_name, limit=limit) for query in tqdm(queries, 'QUERIES', position=0, leave=True)]
        reranked = [self.reranker.rerank(result, queries[i], top_k=top_k) for i, result in enumerate(tqdm(results, 'RERANKING'))]
        return reranked

    async def aget_actual_outputs(self,
                                  user_messages: list[str],
                                  temperature: float=1.0,
                                  max_tokens: int=500,
                                  **kwargs) -> list[str]:
        tasks = [self.llm.achat_completion(huberman_system_message, user_message, temperature=temperature, max_tokens=max_tokens, **kwargs) 
                 for user_message in user_messages]
        responses = await tqdm_asyncio.gather(*tasks, desc='LLM CALLS')
        return responses
    
    async def acreate_test_cases(self,
                                 queries: list[str],
                                 collection_name: str,
                                 limit: int=200,
                                 top_k: int=3
                                 ) -> list[LLMTestCase]:
        '''
        Creates a list of LLM Test Cases based on query retrievals. 
        '''
        reranked_results = self.retrieve_results(queries, collection_name, limit, top_k)
        user_messages = [generate_prompt_series(queries[i], rerank) for i, rerank in enumerate(reranked_results)]
        actual_outputs = await self.aget_actual_outputs(user_messages)
        retrieval_contexts = [create_context_blocks(rerank) for rerank in reranked_results]
        test_cases = [LLMTestCase(input=input, actual_output=output, retrieval_context=context) \
                    for input, output, context in list(zip(queries, actual_outputs, retrieval_contexts))]
        return test_cases

class PollingEvaluation:

    def __init__(self,
                 batch_size: int=10
                ):
        
        if batch_size <= 1:
            raise ValueError("Batch size must be greater than 1")
        self.batch_size = batch_size

    def evaluate_answer_correctness(self,
                                    test_cases: list[LLMTestCase],
                                    model: str | DeepEvalBaseLLM,
                                    show_eval_progress: bool=False,
                                    return_raw: bool=False
                                    ) -> dict[str, Any]:
        '''
        Uses a single model to evaluate the correctness of the LLM generated answers
        over a list of test cases.
        '''
        model_name = model if isinstance(model, str) else model.model
        ac_metric = AnswerCorrectnessMetric(model=model)
        responses = evaluate(test_cases, [ac_metric], print_results=False, show_indicator=show_eval_progress)
        if return_raw:
            return responses
        eval_responses = [load_eval_response(r.metrics[0], r) for r in responses]
        scores = [r.score for r in eval_responses]
        cost = [r.cost for r in eval_responses if r.cost]
        cost = sum(cost) if any(cost) else 'N/A'
        results_dict = {'model': model_name, 'results': eval_responses, 'scores': scores, 'cost': cost}
        return results_dict
    
    def polling_evaluation(self,
                           test_cases: list[LLMTestCase],
                           models: list[str | DeepEvalBaseLLM],
                           show_eval_progress: bool=False,
                           ) -> dict[str,Any]:
        '''
        Iteratively loops through list of models and executes deepeval evaluation function. 
        This function implements "polling evaluation" wherein multiple model scores are 
        crowdsourced vs. an evaluation that uses a single monolithic model. 
        
        Batch size no greater than 10 is recommended to avoid Rate Limit Errors, given that 
        there is no internal backoff/retry support for this function. 
        '''
        test_cases = self._check_test_case_types(test_cases)
        num_batches = ceil(len(test_cases)/self.batch_size)
        model_names = [model if isinstance(model, str) else model.model for model in models]
        results_dict = {name: {'results':[], 'scores': [], 'cost_per_batch': []} for name in model_names}
        for i in range(num_batches):
            batch = test_cases[i*self.batch_size:(i+1)*self.batch_size]
            for model in tqdm(models, desc=f'Batch: {i+1} of {num_batches}'):
                model_name = model if isinstance(model, str) else model.model
                model_results = self.evaluate_answer_correctness(batch, model, show_eval_progress, return_raw=False)
                results_dict[model_name]['results'].extend(model_results['results'])
                results_dict[model_name]['scores'].extend(model_results['scores'])
                results_dict[model_name]['cost_per_batch'].append(model_results['cost'])
                print(f'Completed batch {i+1} results for model: {model_name}')
        model_scores = np.array([results_dict[model_name]['scores'] for model_name in model_names])
        mean_scores = np.mean(model_scores, axis=0)
        evaluation_score = np.mean(mean_scores)
        evaluation_results = {'responses': results_dict, 'mean_scores': mean_scores, 'evaluation_score': round(evaluation_score,3)}
        return evaluation_results
    
    def _check_test_case_types(self, test_cases: list[LLMTestCase]) -> bool:
        if all(isinstance(test_case, LLMTestCase) for test_case in test_cases):
            return test_cases
        if isinstance(test_cases[0], dict):
            try:
                test_cases = [LLMTestCase(**data) for data in test_cases]
                return test_cases
            except Exception as _:
                raise ValueError("Test cases must be a list of LLMTestCase objects or a list of dictionaries containing the keys: ['input', 'actual_output', 'retrieval_context']")




    
# def get_answer_score(query: str,
#                      rag_pipeline: RAGPipeline,
#                      evaluation_llm: str='gpt-4-turbo',
#                      return_context_data: bool=False
#                      ):
#     #define our metric, in this case AnswerCorrectness
#     metric = AnswerCorrectnessMetric(model=evaluation_llm)
    
#     #this is an instance of the RAGPipeline Class
#     data = rag_pipeline(query, verbosity=2)

#     #unpack the results of executing the pipeline
#     query, actual_output, context = data['query'], data['answer'], data['context']

#     #reformat context from list of dicts into list of strings
#     retrieval_context = create_context_blocks(context)
    
#     #define a LLM Test Case on the fly
#     test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=retrieval_context)

#     #execute call to evaluation LLM to evaluate AnswerCorrectness
#     metric.measure(test_case)

#     #return response as an EvalResponse (nothing special just an easier way to organize info
#     response = load_eval_response(metric, test_case, return_context_data=return_context_data)
#     return response

# retrieval_args = list(inspect.signature(client.hybrid_search).parameters)
# retrieval_dict = {k:v for k,v in retrieval_kwargs.items() if k in retrieval_args}
# llm_args = list(inspect.signature(self.llm.chat_completion).parameters)
# llm_dict = {k:v for k,v in llm_kwargs.items() if k in llm_args}