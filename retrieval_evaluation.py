#external files
from openai_interface import GPT_Turbo
from weaviate_interface import WeaviateClient
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from reranker import ReRanker

#standard library imports
import json
import time
import os
import random
from math import ceil
from datetime import datetime
from typing import List, Any, Dict, Tuple, Union, Literal

#misc
from tqdm import tqdm


def sample_data(data: List[dict], sample_size: int):
    sample = random.sample(data, sample_size)
    contents = [(d['doc_id'], d['content']) for d in sample]
    return contents

def get_meta(sample: List[dict], key: str="doc_id") -> List[Any]:
    return [d[key] for d in sample]

def get_sample(doc_id: str, corpus: List[dict], full_dict: bool=False):
    result = [d for d in corpus if d['doc_id'] == doc_id][0]
    if full_dict: return result
    else: return result['content']

def strip_numbers(query: str):
    return query[3:].strip()

def process_questions(question_tuples: List[tuple]) -> Dict[str, List[str]]:
    question_dict = {}
    for tup in question_tuples:
        doc_id = tup[0]
        questions = tup[1].split('\n')
        questions = [strip_numbers(q) for q in questions]
        question_dict[doc_id] = questions
    return question_dict

def generate_dataset(data: List[dict], dir_path: str, num_questions: int=100, batch_size: int=50):
    gpt = GPT_Turbo()
    if batch_size > 50:
        raise ValueError('Due to OpenAI rate limits, batch_size cannot be greater than 50')

    time_marker = datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
    filepath = os.path.join(dir_path, f"{num_questions}_questions_{time_marker}.json")
    
    sample = sample_data(data, num_questions)
    batches = ceil(num_questions/batch_size)
    all_questions = []
    for n in range(batches):
        batch = sample[n*batch_size:(n+1)*batch_size]
        questions = gpt.batch_generate_question_context_pairs(batch)
        all_questions.append(questions)
        if n < batches - 1:
            print('Pausing for 60 seconds due to OpenAI rate limits...')
            time.sleep(60)
    all_questions = [tup for batch in all_questions for tup in batch]
    processed_questions = process_questions(all_questions)
    with open(filepath, 'w') as f:
        json.dump(processed_questions, f, indent=4)
    return processed_questions

def retrieval_evaluation(dataset: EmbeddingQAFinetuneDataset, 
                         class_name: str, 
                         retriever: WeaviateClient,
                         reranker: ReRanker=None,
                         alpha: float=0.5,
                         retrieve_limit: int=5,
                         results_top_k: int=5,
                         rerank_top_k: int=5,
                         chunk_size: int=256,
                         hnsw_config_keys: List[str]=['maxConnections', 'efConstruction', 'ef'],
                         display_properties: List[str]=['doc_id', 'content'],
                         user_def_params: dict=None
                         ) -> Tuple[int, int, int]:
    '''
    Given a dataset, a retriever, and a reranker, evaluate the performance of the retriever and reranker. 
    Returns a dict of kw, vector, and hybrid hit rates and mrr scores. If inlude_miss_info is True, will
    also return a list of kw and vector responses and their associated queries that did not return a hit.

    Args:
    -----
    dataset: EmbeddingQAFinetuneDataset
        Dataset to be used for evaluation
    class_name: str
        Name of Class on Weaviate host to be used for retrieval
    retriever: WeaviateClient
        WeaviateClient object to be used for retrieval 
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
    results_top_k: int=5
        Number of top results to evaluate
    rerank_top_k: int=5
        Number of top results to rerank
    chunk_size: int=256
        Number of tokens used to chunk text
    display_properties: List[str]=['doc_id', 'content']
        List of properties to be returned from Weaviate host for display in response
    '''
    if results_top_k > retrieve_limit:  # we don't want to retrieve less results than the top_k that we want to see returned
        retrieve_limit = results_top_k
        
    reranker_name = reranker.model_name if reranker else "None"
    
    results_dict = {'n':retrieve_limit, 
                    'top_k': results_top_k,
                    'alpha': alpha,
                    'Retriever': retriever.model_name_or_path, 
                    'Ranker': reranker_name,
                    'chunk_size': chunk_size,
                    'kw_hit_rate': 0,
                    'kw_mrr': 0,
                    'vector_hit_rate': 0,
                    'vector_mrr': 0,
                    'hybrid_hit_rate':0,
                    'hybrid_mrr': 0,
                    'total_misses': 0,
                    'total_questions':0
                    }
    #add extra params to results_dict
    results_dict = add_params(retriever, class_name, results_dict, user_def_params, hnsw_config_keys)
    if reranker:
        results_dict['rerank_top_k'] = rerank_top_k  # have to build the results_dict before we can add this information
        
    start = time.perf_counter()
    for query_id, q in tqdm(dataset.queries.items(), 'Queries'):
        results_dict['total_questions'] += 1
        
        #make Keyword, Vector, and Hybrid calls to Weaviate host
        try:
            kw_response = retriever.keyword_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)
            vector_response = retriever.vector_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)
            hybrid_response = retriever.hybrid_search(request=q, class_name=class_name, alpha=alpha, limit=retrieve_limit, display_properties=display_properties)           
            #rerank returned responses if reranker is provided
            if reranker:
                kw_response = reranker.rerank(kw_response, q, top_k=rerank_top_k)
                vector_response = reranker.rerank(vector_response, q, top_k=rerank_top_k)
                hybrid_response = reranker.rerank(hybrid_response, q, top_k=rerank_top_k)
            
            #collect doc_ids to check for document matches (include only results_top_k)
            kw_doc_ids = {result['doc_id']:i for i, result in enumerate(kw_response[:results_top_k], 1)}
            vector_doc_ids = {result['doc_id']:i for i, result in enumerate(vector_response[:results_top_k], 1)}
            hybrid_doc_ids = {result['doc_id']:i for i, result in enumerate(hybrid_response[:results_top_k], 1)}
            
            #extract doc_id for scoring purposes
            doc_id = dataset.relevant_docs[query_id][0]
     
            #increment hit_rate counters and mrr scores
            if doc_id in kw_doc_ids:
                results_dict['kw_hit_rate'] += 1
                results_dict['kw_mrr'] += 1/kw_doc_ids[doc_id]
            if doc_id in vector_doc_ids:
                results_dict['vector_hit_rate'] += 1
                results_dict['vector_mrr'] += 1/vector_doc_ids[doc_id]
            if doc_id in hybrid_doc_ids:
                results_dict['hybrid_hit_rate'] += 1
                results_dict['hybrid_mrr'] += 1/hybrid_doc_ids[doc_id]

            # if no hits, let's capture that
            else:
                results_dict['total_misses'] += 1
                
        except Exception as e:
            print(e)
            continue

    #use raw counts to calculate final scores
    calc_hit_rate_scores(results_dict, search_type='all')
    calc_mrr_scores(results_dict, search_type='all')
    
    end = time.perf_counter() - start
    print(f'Total Processing Time: {round(end/60, 2)} minutes')
    record_results(results_dict, chunk_size, './eval_results', as_text=True)
    return results_dict

def calc_hit_rate_scores(results_dict: Dict[str, Union[str, int]], 
                         search_type: Literal['kw', 'vector', 'hybrid', 'all']=['kw', 'vector']
                         ) -> None:
    if search_type == 'all':
        search_type = ['kw', 'vector', 'hybrid']
    for prefix in search_type:
        results_dict[f'{prefix}_hit_rate'] = round(results_dict[f'{prefix}_hit_rate']/results_dict['total_questions'],2)

def calc_mrr_scores(results_dict: Dict[str, Union[str, int]],
                    search_type: Literal['kw', 'vector', 'hybrid', 'all']=['kw', 'vector']
                    ) -> None:
    if search_type == 'all':
        search_type = ['kw', 'vector', 'hybrid']
    for prefix in search_type:
        results_dict[f'{prefix}_mrr'] = round(results_dict[f'{prefix}_mrr']/results_dict['total_questions'],2)

def create_dir(dir_path: str) -> None:
    '''
    Checks if directory exists, and creates new directory
    if it does not exist
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def record_results(results_dict: Dict[str, Union[str, int]], 
                   chunk_size: int, 
                   dir_outpath: str='./eval_results',
                   as_text: bool=False
                   ) -> None:
    '''
    Write results to output file in either txt or json format

    Args:
    -----
    results_dict: Dict[str, Union[str, int]]
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

def add_params(client: WeaviateClient, 
               class_name: str, 
               results_dict: dict, 
               param_options: dict, 
               hnsw_config_keys: List[str]
              ) -> dict:
    hnsw_params = {k:v for k,v in client.show_class_config(class_name)['vectorIndexConfig'].items() if k in hnsw_config_keys}
    if hnsw_params:
        results_dict = {**results_dict, **hnsw_params}
    if param_options and isinstance(param_options, dict):
        results_dict = {**results_dict, **param_options}
    return results_dict
    




