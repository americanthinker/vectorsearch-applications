#external files
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.llm.llm_interface import LLM
from src.reranker import ReRanker

#standard library imports
import json
import time
import uuid
import os
import re
import random
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
                 llm: LLM
                 ):
        self.llm = llm
        self.reranker = ReRanker()

    def _clean_validate_data(self,
                             data: list[dict], 
                             valid_fields: list[str]=['content', 'summary', 'guest', 'doc_id'],
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

    def _remove_bad_questions(self, questions: list[str]) -> list[str]:
        '''
        Removes questions that contain either the words 'transcript' or 'episode'.
        These questions will potentially add unnecessary noise to the dataset.
        '''
        removal_words = ['transcript', 'episode']
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

def execute_evaluation(dataset: dict, 
                       class_name: str, 
                       retriever: WeaviateWCS,
                       reranker: ReRanker=None,
                       alpha: float=0.5,
                       retrieve_limit: int=100,
                       top_k: int=5,
                       chunk_size: int=256,
                       hnsw_config_keys: list[str]=['maxConnections', 'efConstruction', 'ef'],
                       search_type: Literal['hybrid', 'all']='all',
                       search_properties: list[str]=['content'],
                       display_properties: list[str]=['doc_id', 'content'],
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
    class_name: str
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
    search_properties: list[str]=['content']
        list of properties to be used for search
    display_properties: list[str]=['doc_id', 'content']
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
                    'hybrid_hit_rate':0,
                    'hybrid_mrr': 0,
                    }
    #add extra params to results_dict
    kw_vector_metrics = {
                         'kw_hit_rate': 0,
                         'kw_mrr': 0,
                         'vector_hit_rate': 0,
                         'vector_mrr': 0
                         }
    if search_type == 'all':
        results_dict = {**results_dict, **kw_vector_metrics}
    results_dict['total_questions'] = 0
    results_dict['total_misses'] = 0
    results_dict = add_params(retriever, class_name, results_dict, user_def_params, hnsw_config_keys)
        
    start = time.perf_counter()
    miss_info_list = []
    for query_id, q in tqdm(dataset.queries.items(), 'Queries'):
        results_dict['total_questions'] += 1
        hit = False
        #make Keyword, Vector, and Hybrid calls to Weaviate host
        try:
            hybrid_response = retriever.hybrid_search(request=q, class_name=class_name, properties=search_properties, alpha=alpha, limit=retrieve_limit, display_properties=display_properties)  
            if search_type == 'all':
                kw_response = retriever.keyword_search(request=q, class_name=class_name, properties=search_properties, limit=retrieve_limit, display_properties=display_properties)
                vector_response = retriever.vector_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)         
            #rerank returned responses if reranker is provided
            if reranker:
                hybrid_response = reranker.rerank(hybrid_response, q, top_k=top_k)
                if search_type == 'all':
                    kw_response = reranker.rerank(kw_response, q, top_k=top_k)
                    vector_response = reranker.rerank(vector_response, q, top_k=top_k)
                
            
            #collect doc_ids to check for document matches (include only results_top_k)
            hybrid_doc_ids = {result['doc_id']:i for i, result in enumerate(hybrid_response[:top_k], 1)}
            if search_type == 'all':
                kw_doc_ids = {result['doc_id']:i for i, result in enumerate(kw_response[:top_k], 1)}
                vector_doc_ids = {result['doc_id']:i for i, result in enumerate(vector_response[:top_k], 1)}
            #extract doc_id for scoring purposes
            doc_id = dataset.relevant_docs[query_id][0]
     
            #increment hit_rate counters and mrr scores
            if doc_id in hybrid_doc_ids:
                results_dict['hybrid_hit_rate'] += 1
                results_dict['hybrid_mrr'] += 1/hybrid_doc_ids[doc_id]
                hit = True
            if search_type == 'all':
                if doc_id in kw_doc_ids:
                    results_dict['kw_hit_rate'] += 1
                    results_dict['kw_mrr'] += 1/kw_doc_ids[doc_id]
                    hit = True
                if doc_id in vector_doc_ids:
                    results_dict['vector_hit_rate'] += 1
                    results_dict['vector_mrr'] += 1/vector_doc_ids[doc_id]
                    hit = True
            # if no hits, let's capture that
            if not hit:
                results_dict['total_misses'] += 1
                miss_info = {'query': q, 
                             'answer': dataset.corpus[doc_id],
                             'doc_id': doc_id,
                             'hybrid_response': hybrid_response}
                if search_type == 'all':
                    miss_info = {**miss_info, 
                                 'kw_response': kw_response,
                                 'vector_response': vector_response}
                miss_info_list.append(miss_info)

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
        return results_dict, miss_info
    return results_dict

def calc_hit_rate_scores(results_dict: dict[str, str | int], 
                         search_type: Literal['kw', 'vector', 'hybrid', 'all']='all'
                         ) -> None:
    '''
    Helper function to calculate hit rate scores
    '''
    search_type = ['kw', 'vector', 'hybrid'] if search_type == 'all' else search_type
    for prefix in search_type:
        results_dict[f'{prefix}_hit_rate'] = round(results_dict[f'{prefix}_hit_rate']/results_dict['total_questions'],2)

def calc_mrr_scores(results_dict: dict[str, str | int],
                    search_type: Literal['kw', 'vector', 'hybrid', 'all']='all'
                    ) -> None:
    '''
    Helper function to calculate mrr scores
    '''
    search_type = ['kw', 'vector', 'hybrid'] if search_type == 'all' else search_type
    for prefix in search_type:
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

def add_params(client: WeaviateWCS, 
               class_name: str, 
               results_dict: dict, 
               param_options: dict, 
               hnsw_config_keys: list[str]
              ) -> dict:
    '''
    Helper function that adds parameters to the results_dict:
    - Adds HNSW Index parameters to results_dict
    - Adds optional user-defined parameters to results_dict
    '''
    hnsw_params = {k:v for k,v in client.show_class_config(class_name)['vectorIndexConfig'].items() if k in hnsw_config_keys}
    if hnsw_params:
        results_dict = {**results_dict, **hnsw_params}
    if param_options and isinstance(param_options, dict):
        results_dict = {**results_dict, **param_options}
    return results_dict
    




