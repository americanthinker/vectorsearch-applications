#external files
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.evaluation.eval_prompt_templates import (
    qa_triplet_generation_prompt,
    dataset_generation_prompt,
    qa_validation_prompt,
    qa_flavors
)
from src.llm.llm_utils import get_token_count
from src.llm.llm_interface import LLM
from src.reranker import ReRanker

#standard library imports
import json
import time
import torch.mps
import uuid
import os
import re
import random
from loguru import logger
from typing import Literal
from datetime import datetime
import pandas as pd
import numpy as np
from rich import print

#misc
from tqdm import tqdm


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
                 system_message: str=None,
                 user_message: str=None
                 ):
        self.llm = llm
        self.reranker = ReRanker()
        self.system_message = system_message
        self.user_message = user_message

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
        if total_chars == None or total_chars < 25:
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

    def _remove_bad_questions(self, questions: str | list[str]) -> list[str]:
        '''
        Removes questions that contain either the words 'transcript' or 'episode'.
        These questions will potentially add unnecessary noise to the dataset.
        '''
        removal_words = ['transcript', 'episode', 'excerpt']
        if isinstance(questions, str):
            questions = [questions]
        for i, q in enumerate(questions):
            for word in removal_words:
                finding = re.findall(word, q)
                if finding:
                    questions[i] = ''
        return questions

    def generate_qa_embedding_pairs(self,
                                    data: list[dict],
                                    generate_prompt_tmpl: str,
                                    num_total_questions: int,
                                    num_questions_per_chunk: int = 2,
                                    system_message: str='You are a helpful assistant.',
                                    total_chars: int=None,
                                    threshold: float=None
                                    ) -> dict:
        """
        Generate query/context pairs from a list of documents. The query/context pairs
        can be used for fine-tuning an embedding model using a MultipleNegativesRankingLoss
        or can be used to create an evaluation dataset for retrieval models.
        """

        question_bank = []
        corpus = {}
        queries = {}
        relevant_docs = {}
        clean_data = self._clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        counter = 0

        while len(question_bank) < num_total_questions:
            chunk = clean_data[counter]
            summary = chunk['summary']
            guest = chunk['guest']
            transcript = chunk['content']
            doc_id = chunk['doc_id']
            counter += 1
            assist_message = generate_prompt_tmpl.format(summary=summary,
                                                         guest=guest,
                                                         transcript=transcript,
                                                         num_questions_per_chunk=num_questions_per_chunk)
            try:
                response = self.llm.chat_completion(system_message,
                                                    assist_message,
                                                    temperature=1.0,
                                                    max_tokens=num_questions_per_chunk*50,
                                                    raw_response=False
                                                   )
            except Exception as e:
                print(e)
                continue

            result = response.strip().split("\n")
            questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            questions = self._remove_bad_questions(questions)
            questions = [question for question in questions if len(question) > 0]
            if not any(questions):
                print('No good questions returned')
                continue
            if threshold:
                pairs = [[question,transcript] for question in questions]
                scores = self.reranker.predict(sentences=pairs, activation_fct=self.reranker.activation_fct)
                indexes_to_keep = np.argwhere(scores >= threshold)
                questions = np.array(questions)[indexes_to_keep]
            if any(questions):
                if len(question_bank) < num_total_questions:
                    corpus[doc_id] = transcript
                    for question in questions:
                        if len(question_bank) < num_total_questions:
                            question_bank.append(question)
                            question_id = str(uuid.uuid4())
                            queries[question_id] = question[0] if isinstance(question, (np.ndarray, list)) else question
                            relevant_docs[question_id] = doc_id
            else:
                print('No questions retrieved for this chunk')
            if len(question_bank) % int((num_total_questions * 0.2)) == 0 and len(question_bank) != 0:
                print(f'{len(question_bank)} questions generated')

        # construct dataset
        return dict(queries=queries, corpus=corpus, relevant_docs=relevant_docs)

    def generate_retrieval_dataset( self,
                                    data: list[dict],
                                    num_total_questions: int,
                                    total_chars: int=None,
                                    threshold: float=None
                                    ) -> dict:
        """
        Generate query/context pairs from a list of documents. The query/context pairs
        can be used for fine-tuning an embedding model using a MultipleNegativesRankingLoss
        or can be used to create an evaluation dataset for retrieval models.
        """
        if num_total_questions % 4 != 0:
            raise ValueError('Number of total questions must be divisible by 4')
        quarter = num_total_questions // 4
        question_bank = []
        corpus = {}
        queries = {}
        relevant_docs = {}
        clean_data = self._clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        progress = tqdm(total=num_total_questions, desc='QA Pair Generation')
        flavor_counter = 0
        qa_flavor = 0
        counter = 0
        system_message = self.system_message if self.system_message else 'You are an expert at generating questions from a given text.'
        while len(question_bank) < num_total_questions:
            chunk = clean_data[counter]
            counter += 1
            summary = chunk['summary']
            guest = chunk['guest']
            title = chunk['title']
            transcript = chunk['content']
            doc_id = chunk['doc_id']
            user_message = dataset_generation_prompt.format(guest=guest,
                                                            title=title,
                                                            transcript=transcript,
                                                            qa_flavor=qa_flavors[qa_flavor]
                                                            )
            try:
                response = self.llm.chat_completion(system_message,
                                                    user_message,
                                                    temperature=1.0,
                                                    max_tokens=50,
                                                    raw_response=False
                                                   )
            except Exception as e:
                print(e)
                continue

            result = response.strip()
            # questions = [
            #     re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            # ]
            questions = self._remove_bad_questions(result)
            questions = [question for question in questions if len(question) > 0]
            if not any(questions):
                print('No good questions returned')
                continue
            else:
                question = questions[0]
                valid_system_message = 'You are an expert at determining the quality of questions generated from a given text.'
                valid_user_message = qa_validation_prompt.format(title=title, transcript=transcript, question=question)
                try:
                    valid_response = self.llm.chat_completion(valid_system_message,
                                                              valid_user_message,
                                                              temperature=1.0,
                                                              max_tokens=8,
                                                              raw_response=False
                                                              )
                except Exception as e:
                    print(e)
                    continue
                # logger.info(f'Valid Response: {valid_response}')
                if valid_response.strip() == '1':
                    flavor_counter += 1
                    progress.update(1)
                    if flavor_counter % quarter == 0 and flavor_counter != num_total_questions:
                        qa_flavor += 1
                        logger.info(f'Changing QA Flavor: at count {flavor_counter}, using qa_flavor {qa_flavor}')

                    if len(question_bank) < num_total_questions:
                        corpus[doc_id] = transcript
                        question_bank.append(question)
                        question_id = str(uuid.uuid4())
                        queries[question_id] = question
                        relevant_docs[question_id] = doc_id
                else:
                    print('No questions retrieved for this chunk')
                if len(question_bank) % int((num_total_questions * 0.2)) == 0 and len(question_bank) != 0:
                    print(f'{len(question_bank)} questions generated')

        # construct dataset
        return dict(queries=queries, corpus=corpus, relevant_docs=relevant_docs)

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
        default_system_message = '''
        You are a machine learning expert who specializes in generating datasets for fine-tuning embedding models.\n
        You are particularly adept at generating sentence triplets for use in a Multiple Negatives Ranking Loss function.
        '''
        system_message = self.system_message if self.system_message else default_system_message
        user_message = self.user_message if self.user_message else qa_triplet_generation_prompt
        if user_message != qa_triplet_generation_prompt:
            logger.warning('You are using a custome user message, which will require a "guest" and "transcript" field in your prompt.')
        valid_json_triplet = []
        clean_data = self._clean_validate_data(data, total_chars=total_chars)
        random.shuffle(clean_data)
        counter = 0
        total_token_count = 0
        progress = tqdm(total=num_total_samples, desc='QA Triplets')
        while len(valid_json_triplet) < num_total_samples:
            chunk = clean_data[counter]
            counter += 1
            guest, transcript, doc_id  = chunk['guest'], chunk['content'], chunk['doc_id']
            prompt = user_message.format(guest=guest,transcript=transcript)
            if capture_token_count:
                total_token_count += get_token_count(prompt)
            try:
                response = self.llm.chat_completion(system_message,
                                                    prompt,
                                                    temperature=1.0,
                                                    max_tokens=150,
                                                    raw_response=False,
                                                    response_format={ "type": "json_object" }
                                                    )

                try:
                    loaded = json.loads(response)
                    if self._check_valid_keys(loaded):
                        loaded['anchor'] = transcript
                        loaded['anchor_doc_id'] = doc_id
                        valid_json_triplet.append(loaded)
                        with open(output_path, 'w') as f:
                            json.dump(valid_json_triplet, f, indent=4)
                        progress.update(1)
                except json.JSONDecodeError:
                    logger.error('Response is not JSON')
                    logger.info(response)
                    continue
            except Exception as e:
                logger.info(e)
                continue
        if capture_token_count:
            logger.info(f'Total Token Count: {total_token_count}')
        return valid_json_triplet


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

            # continue

            # result = response.strip().split("\n")
            # questions = [
            #     re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            # ]
            # questions = self._remove_bad_questions(questions)
            # questions = [question for question in questions if len(question) > 0]
            # if not any(questions):
            #     print('No good questions returned')
            #     continue
            # if threshold:
            #     pairs = [[question,transcript] for question in questions]
            #     scores = self.reranker.predict(sentences=pairs, activation_fct=self.reranker.activation_fct)
            #     indexes_to_keep = np.argwhere(scores >= threshold)
            #     questions = np.array(questions)[indexes_to_keep]
            # if any(questions):
            #     if len(question_bank) < num_total_questions:
            #         corpus[doc_id] = transcript
            #         for question in questions:
            #             if len(question_bank) < num_total_questions:
            #                 question_bank.append(question)
            #                 question_id = str(uuid.uuid4())
            #                 queries[question_id] = question[0] if isinstance(question, (np.ndarray, list)) else question
            #                 relevant_docs[question_id] = doc_id
            # else:
            #     print('No questions retrieved for this chunk')
            # if len(question_bank) % int((num_total_questions * 0.2)) == 0 and len(question_bank) != 0:
            #     print(f'{len(question_bank)} questions generated')

        # # construct dataset
        # return dict(queries=queries, corpus=corpus, relevant_docs=relevant_docs)

def execute_evaluation(dataset: dict,
                       collection_name: str,
                       retriever: WeaviateWCS,
                       reranker: ReRanker=None,
                       alpha: float=0.5,
                       retrieve_limit: int=100,
                       top_k: int=5,
                       chunk_size: int=256,
                       search_type: Literal['hybrid', 'kw', 'vector', 'all']=['all'],
                       query_properties: list[str]=['content'],
                       return_properties: list[str]=['doc_id', 'content'],
                       dir_outpath: str='./eval_results',
                       include_miss_info: bool=False,
                       user_def_params: dict=None
                       ) -> dict | tuple[dict, list[dict]]:
    '''
    Given a dataset, a retriever, and a reranker, evaluate the performance of the retriever and reranker.
    Returns a dict of kw, vector, and hybrid hit rates and mrr scores. If inlude_miss_info is True, will
    also return a list of kw and vector responses and their associated queries that did not return a hit.

    Args:
    -----
    dataset: dict
        Dataset to be used for evaluation
    collection_name: str
        Name of Class on Weaviate host to be used for retrieval
    retriever: WeaviateWCS
        WeaviateWCS object to be used for retrieval
    reranker: ReRanker
        ReRanker model to be used for results reranking
    alpha: float=0.5
        Weighting factor for BM25 and Vector search.
        alpha can be any number from 0 to 1, defaulting to 0.5:
            alpha = 0 executes a pure keyword search method (BM25)
            alpha = 0.5 weighs the BM25 and vector methods evenly
            alpha = 1 executes a pure vector search method
    retrieve_limit: int=5
        Number of documents to retrieve from Weaviate host
    top_k: int=5
        Number of top results to evaluate
    chunk_size: int=256
        Number of tokens used to chunk text
    hnsw_config_keys: list[str]=['maxConnections', 'efConstruction', 'ef']
        list of keys to be used for retrieving HNSW Index parameters from Weaviate host
    search_type: Literal['kw', 'vector', 'hybrid', 'all']='all'
        Type of search to be evaluated.  Options are 'kw', 'vector', 'hybrid', or 'all'
    query_properties: list[str]=['content']
        list of properties to be used for search
    return_properties: list[str]=['doc_id', 'content']
        list of properties to be returned from Weaviate host for display in response
    dir_outpath: str='./eval_results'
        Directory path for saving results.  Directory will be created if it does not
        already exist.
    include_miss_info: bool=False
        Option to include queries and their associated search response values
        for queries that are "total misses"
    user_def_params : dict=None
        Option for user to pass in a dictionary of user-defined parameters and their values.
        Will be automatically added to the results_dict if correct type is passed.
    '''
    reranker_name = reranker.model_name if reranker else "None"

    results_dict = {'n':retrieve_limit,
                    'top_k': top_k,
                    'alpha': alpha,
                    'Retriever': retriever.model_name_or_path,
                    'Ranker': reranker_name,
                    'chunk_size': chunk_size,
                    'query_props': query_properties,
                    'total_misses': 0,
                    'total_questions':0
                    }
    search_type = ['kw', 'vector', 'hybrid'] if search_type == ['all'] else search_type
    results_dict = _add_metrics(results_dict, search_type)
    results_dict = add_params(results_dict, user_def_params) if user_def_params else results_dict

    start = time.perf_counter()
    miss_info_list = []
    torch.mps.empty_cache()
    for query_id, q in tqdm(dataset['queries'].items(), 'Queries'):
        results_dict['total_questions'] += 1
        hit = False
        #make Keyword, Vector, and Hybrid calls to Weaviate host
        try:
            if 'hybrid' in search_type:
                hybrid_doc_ids,hybrid_response = get_doc_ids('hybrid', retriever, q, collection_name, reranker, return_properties,
                                                 retrieve_limit, top_k, alpha, query_properties)
            if 'kw' in search_type:
                kw_doc_ids,kw_response = get_doc_ids('kw', retriever, q, collection_name, reranker, return_properties,
                                         retrieve_limit, top_k, query_properties=query_properties)
            if 'vector' in search_type:
                vector_doc_ids,vector_response = get_doc_ids('vector', retriever, q, collection_name, reranker,
                                                 return_properties,retrieve_limit, top_k)

            #extract doc_id for scoring purposes
            doc_id = dataset['relevant_docs'][query_id]

            #increment hit_rate counters and mrr scores
            if 'hybrid_doc_ids' in locals():
                if doc_id in hybrid_doc_ids:
                    results_dict['hybrid_hit_rate'] += 1
                    results_dict['hybrid_mrr'] += 1/hybrid_doc_ids[doc_id]
                    hit = True
            if 'kw_doc_ids' in locals():
                if doc_id in kw_doc_ids:
                    results_dict['kw_hit_rate'] += 1
                    results_dict['kw_mrr'] += 1/kw_doc_ids[doc_id]
                    hit = True
            if 'vector_doc_ids' in locals():
                if doc_id in vector_doc_ids:
                    results_dict['vector_hit_rate'] += 1
                    results_dict['vector_mrr'] += 1/vector_doc_ids[doc_id]
                    hit = True

            # if no hits, let's capture that
            if not hit:
                results_dict['total_misses'] += 1
                response_misses = []
                if 'hybrid_response' in locals():
                    hybrid_miss_info = _create_miss_info('hybrid', q, hybrid_response, dataset, doc_id, reranker=reranker)
                    response_misses.append(hybrid_miss_info)
                if 'kw_response' in locals():
                    kw_miss_info = _create_miss_info('kw', q, kw_response, dataset, doc_id, reranker=reranker)
                    response_misses.append(kw_miss_info)
                if 'vector_response' in locals():
                    vector_miss_info = _create_miss_info('vector', q, vector_response, dataset, doc_id, reranker=reranker)
                    response_misses.append(vector_miss_info)
                miss_info_list.append(response_misses)

        except Exception as e:
            print(e)
            continue

    #use raw counts to calculate final scores
    calc_hit_rate_scores(results_dict, search_type=search_type)
    calc_mrr_scores(results_dict, search_type=search_type)

    end = time.perf_counter() - start
    print(f'Total Processing Time: {round(end/60, 2)} minutes')
    record_results(results_dict, chunk_size, dir_outpath=dir_outpath, as_text=True)

    if include_miss_info:
        return results_dict, miss_info_list
    return results_dict

def calc_hit_rate_scores(results_dict: dict[str, str | int],
                         search_type: Literal['kw', 'vector', 'hybrid', 'all']=['all']
                         ) -> None:
    '''
    Helper function to calculate hit rate scores
    '''
    accepted_search_types = ['kw', 'vector', 'hybrid']
    _check_search_type_param(search_type)
    search_type = ['kw', 'vector', 'hybrid'] if search_type == ['all'] else search_type
    for prefix in search_type:
        if prefix in accepted_search_types:
            results_dict[f'{prefix}_hit_rate'] = round(results_dict[f'{prefix}_hit_rate']/results_dict['total_questions'],2)

def calc_mrr_scores(results_dict: dict[str, str | int],
                    search_type: Literal['kw', 'vector', 'hybrid', 'all']=['all']
                    ) -> None:
    '''
    Helper function to calculate mrr scores
    '''
    accepted_search_types = ['kw', 'vector', 'hybrid']
    _check_search_type_param(search_type)
    search_type = accepted_search_types if search_type == ['all'] else search_type
    for prefix in search_type:
        if prefix in accepted_search_types:
            results_dict[f'{prefix}_mrr'] = round(results_dict[f'{prefix}_mrr']/results_dict['total_questions'],2)

def create_dir(dir_path: str) -> None:
    '''
    Checks if directory exists, and creates new directory
    if it does not exist
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def record_results(results_dict: dict[str, str | int],
                   chunk_size: int,
                   dir_outpath: str='./eval_results',
                   as_text: bool=False
                   ) -> None:
    '''
    Write results to output file in either txt or json format

    Args:
    -----
    results_dict: dict[str, str | int]
        Dictionary containing results of evaluation
    chunk_size: int
        Size of text chunks in tokens
    dir_outpath: str
        Path to output directory.  Directory only, filename is hardcoded
        as part of this function.
    as_text: bool
        If True, write results as text file.  If False, write as json file.
    '''
    create_dir(dir_outpath)
    time_marker = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ext = 'txt' if as_text else 'json'
    path = os.path.join(dir_outpath, f'retrieval_eval_{chunk_size}_{time_marker}.{ext}')
    if as_text:
        with open(path, 'a') as f:
            f.write(f"{results_dict}\n")
    else:
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=4)

def get_doc_ids(search_mode: str,
                retriever: WeaviateWCS,
                query: str,
                collection_name: str,
                reranker: ReRanker,
                return_properties: list[str],
                retrieve_limit: int,
                top_k: int,
                alpha: float=None,
                query_properties: list[str]=None
                ) -> list[str]:
    if search_mode == 'hybrid':
        response = retriever.hybrid_search(request=query, collection_name=collection_name, query_properties=query_properties,
                                           alpha=alpha,limit=retrieve_limit,return_properties=return_properties)
    elif search_mode == 'kw':
        response = retriever.keyword_search(request=query, collection_name=collection_name, query_properties=query_properties,
                                            limit=retrieve_limit, return_properties=return_properties)
    elif search_mode == 'vector':
        response = retriever.vector_search(request=query, collection_name=collection_name, limit=retrieve_limit,
                                           return_properties=return_properties)
    if reranker:
        response = reranker.rerank(response, query, top_k=top_k)
    doc_ids = {result['doc_id']:i for i, result in enumerate(response[:top_k], 1)}
    return doc_ids, response

def _check_search_type_param(search_type: list) -> None:
    accepted_search_types = ['kw', 'vector', 'hybrid', 'all']
    if not isinstance(search_type, list):
        raise ValueError(f'search_type must be a list, received a {type(search_type)}')
    if not any(search_type):
        raise ValueError(f'search_type must contain at least one search type from {accepted_search_types}')
    count = 0
    for search_type_ in search_type:
        if search_type_ in accepted_search_types:
            count += 1
    if count == 0:
        raise ValueError(f'Please use one of {accepted_search_types}. Received {search_type}')

def _add_metrics(results_dict: dict,
                 search_type: Literal['kw', 'vector', 'hybrid', 'all']=['all']
                 ) -> dict:
    '''
    Helper function to add metrics to results_dict
    '''
    accepted_search_types = ['kw', 'vector', 'hybrid']
    _check_search_type_param(search_type)
    search_type = ['kw', 'vector', 'hybrid'] if search_type == ['all'] else search_type
    for prefix in search_type:
        if prefix in accepted_search_types:
            results_dict = {**results_dict,
                            f'{prefix}_hit_rate': 0,
                            f'{prefix}_mrr': 0}
    return results_dict

def _create_miss_info(search_type: str,
                      query: str,
                      response: list[dict],
                      dataset: dict,
                      doc_id: str,
                      reranker=None) -> dict:
    '''
    Creates miss_info dict for queries that do not return a hit
    '''
    miss_info = {'query': query,
                 'answer': dataset['corpus'][doc_id],
                 'answer_doc_id': doc_id,
                 f'{search_type}_response': response}
    if reranker:
        miss_info['query_answer_cross_score'] = reranker.cross_enc_score(query, dataset['corpus'][doc_id])
    return miss_info

def add_params(results_dict: dict,
               param_options: dict,
              ) -> dict:
    '''
    Helper function that adds parameters to the results_dict:
    - Adds HNSW Index parameters to results_dict
    - Adds optional user-defined parameters to results_dict
    '''
    # hnsw_params = {k:v for k,v in client.show_class_config(collection_name)['vectorIndexConfig'].items() if k in hnsw_config_keys}
    # if hnsw_params:
    #     results_dict = {**results_dict, **hnsw_params}
    if param_options and isinstance(param_options, dict):
        results_dict = {**results_dict, **param_options}
    return results_dict





