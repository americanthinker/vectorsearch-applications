#external files
from openai_interface import GPT_Turbo
from weaviate_interface import WeaviateClient
from reranker import ReRanker

#standard library imports
import json
import time
import os
import random
from math import ceil
from datetime import datetime
from typing import List, Any, Dict, Tuple, Union

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

def execute_evaluation( dataset: Dict[str, List[str]], 
                        retriever: WeaviateClient,
                        reranker: ReRanker,
                        class_name: str, 
                        alpha: float=0.5,
                        limit: int=10,
                        top_k: int=5,
                        chunk_size: int=256,
                        rerank_all_responses: bool=False,
                        ) -> Tuple[int, int, int, int]:

    top_k = top_k if top_k else limit
    reranker_name = reranker.model_name if rerank_all_responses else "None"
    
    results_dict = {'n':limit, 
                    'top_k': top_k, 
                    'alpha': alpha,
                    'Retriever': retriever.model_name_or_path, 
                    'Ranker': reranker_name,
                    'chunk_size': chunk_size,
                    'kw_hit_rate': 0,
                    'vector_hit_rate': 0,
                    'hybrid_hit_rate':0,
                    'combined_hit_rate': 0,
                    'total_questions':0
                    }
    for doc_id, questions in tqdm(dataset.items(), 'Questions'):
        for q in questions:
            results_dict['total_questions'] += 1
            
            #make calls to Weaviate host: Keyword, Vector, and Hybrid
            try:
                kw_response = retriever.keyword_search(query=q, class_name=class_name, limit=limit)
                vector_response = retriever.vector_search(query=q, class_name=class_name, limit=limit)
                weaviate_hybrid_response = retriever.hybrid_search(query=q, class_name=class_name, alpha=alpha, limit=limit)
                combined_hybrid_response = kw_response + vector_response                
            
                #rerank returned responses if rerank_all is True
                if rerank_all_responses:
                    kw_response = reranker.rerank(kw_response, q, top_k=top_k)
                    vector_response = reranker.rerank(vector_response, q, top_k=top_k)
                    weaviate_hybrid_response = reranker.rerank(weaviate_hybrid_response, q, top_k=top_k)
                    combined_hybrid_response = reranker.rerank(combined_hybrid_response, q, top_k=top_k)
                
                #collect doc_ids to check for document matches (include only top_k if top_k > 0)
                kw_doc_ids = [res['doc_id'] for res in kw_response][:top_k]
                vector_doc_ids = [res['doc_id'] for res in vector_response][:top_k]
                hybrid_doc_ids = [res['doc_id'] for res in weaviate_hybrid_response][:top_k]
                combined_doc_ids = [res['doc_id'] for res in combined_hybrid_response][:top_k]
                
                #increment hit_rate counters as appropriate
                if doc_id in kw_doc_ids:
                    results_dict['kw_hit_rate'] += 1
                if doc_id in vector_doc_ids:
                    results_dict['vector_hit_rate'] += 1
                if doc_id in hybrid_doc_ids:
                    results_dict['hybrid_hit_rate'] += 1
                if doc_id in combined_doc_ids:
                    results_dict['combined_hit_rate'] += 1
                    
            except Exception as e:
                print(e)
                continue

    #use raw counts to calculate final scores
    calc_hit_rate_scores(results_dict)
    
    return results_dict

def calc_hit_rate_scores(results_dict: Dict[str, Union[str, int]]) -> None:
    for prefix in ['kw', 'vector', 'hybrid']:
        results_dict[f'{prefix}_hit_rate'] = round(results_dict[f'{prefix}_hit_rate']/results_dict['total_questions'],2)

def calc_mrr_scores(results_dict: Dict[str, Union[str, int]]) -> None:
    for prefix in ['kw', 'vector', 'hybrid']:
        results_dict[f'{prefix}_mrr'] = round(results_dict[f'{prefix}_mrr']/results_dict['total_questions'],2)

def record_results(results_dict: Dict[str, Union[str, int]], 
                   chunk_size: int, 
                   dir_outpath: str,
                   as_text: bool=False
                   ) -> None:
    '''
    Write results to output file

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
    time_marker = datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
    path = os.path.join(dir_outpath, f'retrieval_eval_{chunk_size}_{time_marker}')
    if as_text:
        path = path + '.txt'
        with open(path, 'a') as f:
            f.write(f"{results_dict}\n")
    else: 
        path = path + '.json'
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=4)




