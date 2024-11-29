#external files
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.evaluation.eval_prompt_templates import (
    QAGenerationResponse,
    QAValidationResponse,
    qa_triplet_generation_prompt,
    dataset_generation_system_prompt,
    dataset_generation_user_prompt2,
    qa_validation_user_prompt,
    qa_validation_system_prompt
)
from src.llm.llm_interface import LLM
from src.reranker import ReRanker
from instructor import from_litellm
from litellm import completion, ModelResponse

#standard library imports
import json
import time
import uuid
import os
import re
import random
from loguru import logger
from typing import Literal
from datetime import datetime
import pandas as pd
from rich import print
from math import ceil

#misc
from tqdm import tqdm
from src.data_models import (
    SearchTypeEnum, 
    GenerationEvaluation,
    RetrievalEvaluation, 
    EvaluationDataset
)
from src.preprocessor.preprocessing import FileIO
from pydantic import BaseModel

class QueryContextGenerator:
    '''
    Class designed for the generation of query/context pairs using a
    Generative LLM. The LLM is used to generate questions from a given
    corpus of text. The query/context pairs can be used to fine-tune 
    an embedding model using a MultipleNegativesRankingLoss loss function
    or can be used to create evaluation datasets for retrieval models.

    Args:
    -----
    llm: LLM
        LLM object used for generating questions from a given corpus
    '''
    def __init__(self, 
                 llm: LLM,
                 use_instructor: bool=False
                 ):
        self.llm = llm 
        self.model_name = llm.model_name
        self.llm = from_litellm(completion) if use_instructor else self.llm
        self.reranker = ReRanker()

    def _clean_validate_data(self,
                             data: list[dict], 
                             valid_fields: list[str]=['content', 'summary', 'guest', 'doc_id', 'title'],
                             total_chars: int=None
                             #TODO: Use HF datasets as return type for data
                             ) -> list[dict]:
        '''
        Strip original data chunks so they only contain valid_fields.
        Remove any chunks less than total_chars in size. Prevents LLM
        from asking questions from sparse content. 
        '''
        if total_chars is None or total_chars < 25:
            contents = [d['content'] for d in data]
            lengths = list(map(len, contents))
            analysis = pd.DataFrame(lengths)
            total_chars = int(analysis.describe().loc['50%'][0])
            print(f'Using a total_chars length of {total_chars}')
        clean_docs = [{k:v for k,v in d.items() if k in valid_fields} for d in data]
        valid_docs = [d for d in clean_docs if len(d['content']) > total_chars]
        return valid_docs

    def train_val_split(self,
                        data: list[dict],
                        n_train_questions: int, 
                        n_val_questions: int, 
                        n_questions_per_chunk: int=2,
                        total_chars: int=None
                        ) -> tuple[list[dict], list[dict]]:
        '''
        Splits corpus into training and validation sets.  Training and 
        validation samples are randomly selected from the corpus. total_chars
        parameter is set based on pre-analysis of average doc length in the 
        training corpus. 
        '''
        clean_data = self._clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        train_index = n_train_questions//n_questions_per_chunk
        valid_index = n_val_questions//n_questions_per_chunk
        end_index = valid_index + train_index
        if end_index > len(clean_data):
            raise ValueError('Cannot create dataset with desired number of questions, try using a larger dataset')
        train_data = clean_data[:train_index]
        valid_data = clean_data[train_index:end_index]
        print(f'Length Training Data: {len(train_data)}')
        print(f'Length Validation Data: {len(valid_data)}')
        return train_data, valid_data

    def _remove_bad_questions(self, question: str) -> str:
        '''
        Removes questions that contain either the words 'transcript' or 'episode'.
        These questions will potentially add unnecessary noise to the dataset.
        '''
        removal_words = ['transcript', 'episode', 'excerpt']
        for word in removal_words:
            finding = re.findall(word, question)
            if finding:
                question = ''
        return question
    
    def _instructor_llm_call(self, 
                            system_message: str, 
                            user_message: str,  
                            response_model: BaseModel,
                            temperature: float=1.0, 
                            **kwargs
                            ) -> dict:
        '''
        Base LLM call for use throughout codebase. 
        '''
        response = self.llm.chat.completions.create(
                                            model=self.model_name,
                                            temperature=temperature,
                                            messages=[
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ],
                    response_model=response_model,
                    **kwargs
                )
        return response
    
    def generate_retrieval_dataset(self,
                                   data: list[dict],
                                   num_total_questions: int,
                                   total_chars: int=None,
                                   threshold: float=None,
                                   return_validations: bool=True
                                   ) -> EvaluationDataset | dict[EvaluationDataset, list[dict[str, str]]]:
        """
        Generate query/context pairs from a list of documents. The query/context pairs
        can be used for fine-tuning an embedding model using a MultipleNegativesRankingLoss
        or can be used to create an evaluation dataset for retrieval models.
        """

        question_bank = []
        validation_errors = []
        corpus = {}
        queries = {}
        relevant_docs = {}
        answers = {}
        clean_data = self._clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        counter = 0
        quartile = ceil(num_total_questions/4)
        progress = tqdm(total=num_total_questions, desc='QA Pairs')

        while len(question_bank) < num_total_questions:
            chunk = clean_data[counter]
            # title = chunk['title']
            # summary = chunk['summary']
            # guest = chunk['guest']
            transcript = chunk['content']
            doc_id = chunk['doc_id']
            counter += 1
            try:
                qa_user_message = dataset_generation_user_prompt2.format(transcript=transcript)
                response: QAGenerationResponse = self._instructor_llm_call(dataset_generation_system_prompt, 
                                               qa_user_message, 
                                               temperature=1.0, 
                                               response_model=QAGenerationResponse)

            except Exception as e:
                logger.info(f'Error captured as: {e}')
                continue
            question = response.question
            answer = response.answer
            question: str = self._remove_bad_questions(question)
            if not any(question):
                logger.info('No good questions returned')
                continue
            user_validation_message = qa_validation_user_prompt.format(transcript=transcript,
                                                                     question=question,
                                                                     answer=answer)
            validation_response: QAValidationResponse = self._instructor_llm_call(qa_validation_system_prompt,
                                                                            user_validation_message,
                                                                            temperature=1.0,
                                                                            response_model=QAValidationResponse)
            if int(validation_response.validation) == 0:
                validation_errors.append({'question': question, 'answer': answer, 'transcript': transcript, 'reasoning': validation_response.reasoning})
                logger.info('Invlaid Question/Answer Pair')
                continue
            if threshold and len(question_bank) == quartile:
                logger.info('Converting threshold to 0')
                threshold = 0
            if threshold:
                pair = [question, transcript]
                score = self.reranker.predict(sentences=pair, activation_fct=self.reranker.activation_fct)
                if score < threshold:
                    logger.info(f'Reranker score too low --> Score: {score}')
                    continue
            if len(question_bank) < num_total_questions:
                progress.update(1)        
                question_id = str(uuid.uuid4())
                queries[question_id] = question
                question_bank.append(question)
                corpus[doc_id] = transcript
                relevant_docs[question_id] = doc_id
                answers[question_id] = answer
                
        # construct dataset
        dataset = EvaluationDataset(queries=queries, corpus=corpus, relevant_docs=relevant_docs, answers=answers)
        if return_validations:
            return {'dataset': dataset, 'validation_errors': validation_errors}
        return dataset

    def generate_qa_triplets(self,
                             data: list[dict],
                             num_total_samples: int,
                             output_path: str=None,
                             total_chars: int=None,
                             capture_token_count: bool=True
                             ) -> dict:
        """
        Generate query/context pairs from a list of documents. The query/context pairs
        can be used for fine-tuning an embedding model using a MultipleNegativesRankingLoss
        or can be used to create an evaluation dataset for retrieval models.
        """
        output_path = output_path if output_path else './qa_triplets.json'
        triplet_system_message = '''
        You are a machine learning expert who specializes in generating datasets for fine-tuning embedding models.\n
        You are particularly adept at generating sentence triplets for use in a Multiple Negatives Ranking Loss function.
        '''
        valid_json_triplets = []
        clean_data = self._clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        counter = 0
        total_token_count = 0
        progress = tqdm(total=num_total_samples, desc='QA Triplets')
        while len(valid_json_triplets) < num_total_samples:
            chunk = clean_data[counter]
            counter += 1
            guest, transcript, doc_id  = chunk['guest'], chunk['content'], chunk['doc_id']
            user_message = qa_triplet_generation_prompt.format(guest=guest,transcript=transcript)
            try:
                response: str | ModelResponse =  self.llm.chat_completion(  triplet_system_message, 
                                                                            user_message,
                                                                            temperature=1.0, 
                                                                            max_tokens=150,
                                                                            raw_response=capture_token_count,
                                                                            response_format={ "type": "json_object" }
                                                                            )
                
                try:
                    if capture_token_count:
                        total_token_count += response.usage.total_tokens
                        loaded = json.loads(response.choices[0].message.content)
                    else:
                        loaded = json.loads(response)
                    if self._check_valid_keys(loaded):
                        loaded['anchor'] = transcript
                        loaded['anchor_doc_id'] = doc_id
                        valid_json_triplets.append(loaded)
                        with open(output_path, 'w') as f:
                            json.dump(valid_json_triplets, f, indent=4)
                        progress.update(1)
                except json.JSONDecodeError:
                    logger.error('Response is not valid JSON')
                    logger.info(response)
                    continue
            except Exception as e:
                logger.info(f'Error with initial LLM call due to {e}')
                continue
        if capture_token_count:
            logger.info(f'Total Token Count: {total_token_count}')
        return valid_json_triplets
    

    def _check_valid_keys(self, 
                          sample: dict, 
                          valid_keys: list[str]=['positive', 'hard_negative']
                          ) -> list[dict]:
        if len(sample) != 2:
            return False 
        for key in valid_keys:
            if key not in sample.keys():
                return False
        return True
    

class RetrievalEvaluationService:
    """Service for evaluating the Retrieval System."""

    accepted_search_types = [member.value for member in SearchTypeEnum]
    keyword = SearchTypeEnum.keyword.value
    vector = SearchTypeEnum.vector.value
    hybrid = SearchTypeEnum.hybrid.value

    def __init__(self, retriever: WeaviateWCS):
        self.retriever = retriever

    def execute_evaluation(
        self,
        dataset: dict[dict[str, str]],
        collection_name: str,
        reranker: ReRanker | None = None,
        alpha: float | None = None,
        retrieve_limit: int = 100,
        top_k: int = 5,
        chunk_size: int = 256,
        chunk_overlap: int = 0,
        query_properties: list[str]=['content'],
        dir_outpath: str = './eval_results',
        include_miss_info: bool = False,
        user_def_params: dict | None = None,
        search_type: Literal['kw', 'vector', 'hybrid', 'all'] = 'all',
    ) -> dict | tuple[dict, list[dict]]:
        """Given a dataset, a retriever, and a reranker, evaluate the performance of the retriever and reranker.

        Returns a RetrievalEvaluation Class of kw, vector, and hybrid hit rates and mrr scores. If inlude_miss_info
        is True, will also return a list of responses and their associated queries that did not return a hit.

        Args:
            dataset: EvaluationDataset
                Golden dataset previously generated to be used for evaluation
            collection_name: str
                Name of the collection to be used for search
            retrieve_limit: int=100
                Number of chunks to retrieve from host DB
            rerank: bool = True
                If True, the chunks will be reranked. If False, they will not
            top_k: int=5
                Number of top results to evaluate for each hit
            chunk_size: int=256
                Number of tokens used to chunk text
            chunk_overlap: int=0
                Number of tokens to overlap between chunks
            alpha: Optional[float]=None
                Weighting factor for FTS and Vector search.
                alpha can be any number from 0 to 1, defaulting to 0.5:
                    alpha = 0 executes a pure keyword search method (BM25)
                    alpha = 0.5 weighs the BM25 and vector methods evenly
                    alpha = 1 executes a pure vector search method
            dir_outpath: str='./eval_results'
                Directory path for saving results.  Directory will be created if it does not
                already exist.
            include_miss_info: bool=False
                Option to include queries and their associated search response values
                for queries that are "total misses"
            user_def_params : dict=None
                Option for user to pass in a dictionary of user-defined parameters and their values.
                Will be automatically added to the RetrievalEvaluation.
            search_type: Literal['kw', 'vector', 'hybrid', 'all']
                What type of search to evaluate
        """
        if search_type not in self.accepted_search_types + ['all']:
            raise ValueError(f'Invalid search type. Must be one of {self.accepted_search_types + ["all"]}')
        reranker_name = reranker.model_name if reranker else "None"

        results = RetrievalEvaluation(
            retrieve_limit=retrieve_limit,
            top_k=top_k,
            retriever=self.retriever.model_name_or_path,
            reranker=reranker_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            search_type=search_type,
            total_misses=0,
            total_questions=0,
            alpha=alpha,
        )
        # add dynamic metrics and parameters to results model
        search_type = self.accepted_search_types if search_type == 'all' else [search_type]
        self._add_metrics(results, search_type)
        if user_def_params:
            self._add_params(results, user_def_params)

        start = time.perf_counter()
        miss_info_list = list()
        return_properties = ['doc_id', 'content']
        for query_id, query in tqdm(dataset['queries'].items(), 'Queries'):
            results.total_questions += 1
            hit = False

            # extract doc_id for scoring purposes
            doc_id = dataset['relevant_docs'][query_id]
            # ensure that alpha value is not None
            alpha = alpha if alpha is not None else 0.5
            # make Keyword, Vector, and Hybrid calls to host DB
            try:
                if self.hybrid in search_type:
                    hybrid_response = self.retriever.hybrid_search(query, collection_name, query_properties, alpha, retrieve_limit, return_properties=return_properties)
                    hybrid_response = (
                        reranker.rerank(hybrid_response, query, top_k)
                        if reranker else hybrid_response[:top_k]
                    )
                    hybrid_doc_ids = {resp['doc_id']: i for i, resp in enumerate(hybrid_response, 1)}
                    if doc_id in hybrid_doc_ids:
                        results.hybrid_raw_hits += 1
                        results.hybrid_mrr += (1 / hybrid_doc_ids[doc_id])
                        hit = True
                if self.keyword in search_type:
                    kw_response = self.retriever.keyword_search(query, collection_name, query_properties, retrieve_limit, return_properties=return_properties)
                    kw_response = (
                        reranker.rerank(kw_response, query, top_k)
                        if reranker else kw_response[:top_k]
                    )
                    kw_doc_ids = {resp['doc_id']: i for i, resp in enumerate(kw_response, 1)}
                    if doc_id in kw_doc_ids:
                        results.kw_raw_hits += 1
                        results.kw_mrr += 1 / kw_doc_ids[doc_id]
                        hit = True
                if self.vector in search_type:
                    vector_response = self.retriever.vector_search(query, collection_name,retrieve_limit, return_properties=return_properties)
                    vector_response = (
                        reranker.rerank(vector_response, query, top_k)
                        if reranker else vector_response[:top_k]
                    )
                    vector_doc_ids = {resp['doc_id']: i for i, resp in enumerate(vector_response, 1)}
                    if doc_id in vector_doc_ids:
                        results.vector_raw_hits += 1
                        results.vector_mrr += 1 / vector_doc_ids[doc_id]
                        hit = True
            except Exception as e:
                print(f'Error captured as: {e}')
                continue

            # if no hits, let's capture that
            if not hit:
                results.total_misses += 1
                miss_info_list.append(doc_id)

        # use raw counts to calculate final scores
        calc_hit_rate_scores(results, search_type=search_type)
        calc_mrr_scores(results, search_type=search_type)

        end = time.perf_counter() - start
        evaluation_time = f'{round(end/60, 2)} minutes'
        logger.info(f'Total Evaluation Time: {evaluation_time}')
        try:
            self._record_results(results, chunk_size, dir_outpath=dir_outpath)
        except Exception as e:
            logger.info(e)
            pass
        if include_miss_info:
            results.miss_info = miss_info_list
        results.evaluation_time = evaluation_time
        return results

    def _add_metrics(self, results_model: RetrievalEvaluation, search_type: list[str]) -> None:
        """Helper function to add metrics to results."""
        for prefix in search_type:
            hits = f'{prefix}_raw_hits'
            mrr_metric = f'{prefix}_mrr'
            setattr(results_model, hits, 0)
            setattr(results_model, mrr_metric, 0)

    def _add_params(
        self,
        results: RetrievalEvaluation,
        param_options: dict,
    ) -> None:
        """Helper function that adds parameters to the RetrievalEvaluation model.

        Adds optional user-defined parameters
        """
        if param_options and isinstance(param_options, dict):
            for k, v in param_options.items():
                setattr(results, k, v)


def calc_hit_rate_scores(
    results: RetrievalEvaluation | dict, search_type: Literal['kw', 'vector', 'hybrid', 'all'] = ['all']
) -> None:
    """Helper function to calculate hit rate scores."""
    search_type = [member.value for member in SearchTypeEnum] if search_type == ['all'] else search_type
    for prefix in search_type:
        key = f'{prefix}_raw_hits'
        raw_hits = getattr(results, key) if isinstance(results, RetrievalEvaluation) else results[key]
        if isinstance(results, RetrievalEvaluation):
            setattr(results, f'{prefix}_hit_rate', round(raw_hits / results.total_questions, 2))
        else: 
            results[f'{prefix}_hit_rate'] = round(raw_hits / results['total_questions'], 2)

def calc_mrr_scores(
    results: RetrievalEvaluation | dict, search_type: Literal['kw', 'vector', 'hybrid', 'all'] = ['all']
) -> None:
    """Helper function to calculate mrr scores."""
    search_type = [member.value for member in SearchTypeEnum] if search_type == ['all'] else search_type
    for prefix in search_type:
        key = f'{prefix}_mrr'
        mrr_value = getattr(results, key) if isinstance(results, RetrievalEvaluation) else results[key]
        if isinstance(results, RetrievalEvaluation):
            setattr(results, key, round(mrr_value / results.total_questions, 2))
        else: 
            results[key] = round(mrr_value / results['total_questions'], 2)

def create_dir(dir_path: str) -> None:
    """Creates new directory if it does not exist, including intermediate directories if necessary."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def record_results(
        results: RetrievalEvaluation | GenerationEvaluation | dict,
        chunk_size: int | None = None,
        dir_outpath: str = './eval_results',
    ) -> None:
        """Write results to output file in either txt or json format.

        Args:
        -----
        results: dict[str, str | int]
            Dictionary containing results of evaluation
        chunk_size: int
            Size of text chunks in tokens
        dir_outpath: str
            Path to output directory.  Directory only, filename is hardcoded
            as part of this function.
        as_text: bool
            If True, write results as text file.  If False, write as json file.
        """
        create_dir(dir_outpath)
        time_marker = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if isinstance(results, dict):
            if not chunk_size:
                raise ValueError('Chunk size must be provided for retrieval evaluation')
            path = os.path.join(dir_outpath, f'retrieval_eval_{chunk_size}_{time_marker}.json')
        else:
            # at this point results is assumed to be a RetrievalEvaluation or GenerationEvaluation object
            results: dict = results.model_dump()
            generation_key = 'reader_model'
            # this is a hack to determine if we are dealing with a generation or retrieval evaluation
            # if 'reader_model' is in the results, then we are dealing with a generation evaluation
            # would prefer to use isinstance, but custome Classes are not being recognized 
            if generation_key in results.keys():
                path = os.path.join(dir_outpath, f'generation_eval_{time_marker}.json')
            else:
                if not chunk_size:
                    raise ValueError('Chunk size must be provided for retrieval evaluation')
                path = os.path.join(dir_outpath, f'retrieval_eval_{chunk_size}_{time_marker}.json')
        FileIO.save_as_json(path, results)