#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[26]:


from preprocessing import FileIO
from typing import List, Any, Dict, Tuple, Union
from llama_index.evaluation import generate_question_context_pairs, generate_qa_embedding_pairs
from openai_interface import GPT_Turbo
import openai
from opensearch_interface import OpenSearchClient
from reranker import ReRanker
from index_templates import youtube_body
import json
import time
import os
from math import ceil
from tqdm import tqdm
from datetime import datetime


# ### Ingest data

# In[3]:


data_path = './practice_data/impact_theory_minilm_196.parquet'
data = FileIO().load_parquet(data_path)


# ### Randomly select 100 chunks for Q/A pairs

# In[4]:


import random


# In[5]:


def sample_data(data: List[dict], sample_size: int):
    sample = random.sample(data, sample_size)
    contents = [(d['doc_id'], d['content']) for d in sample]
    return contents


# In[6]:


def get_meta(sample: List[dict], key: str="doc_id") -> List[Any]:
    return [d[key] for d in sample]


# In[7]:


def get_sample(doc_id: str, corpus: List[dict], full_dict: bool=False):
    result = [d for d in corpus if d['doc_id'] == doc_id][0]
    if full_dict: return result
    else: return result['content']


# In[8]:


get_sample('kE3yryW-FiE_33', data)


# In[9]:


def strip_numbers(query: str):
    return query[3:].strip()


# In[10]:


def process_questions(question_tuples: List[tuple]) -> Dict[str, List[str]]:
    question_dict = {}
    for tup in question_tuples:
        doc_id = tup[0]
        questions = tup[1].split('\n')
        questions = [strip_numbers(q) for q in questions]
        question_dict[doc_id] = questions
    return question_dict


# In[11]:


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


# In[12]:


dataset = generate_dataset(data=data, dir_path='./practice_data/', num_questions=100)


# In[89]:


gteclient = OpenSearchClient(model_name_or_path='/home/elastic/notebooks/vector_search_applications/models/gte-base/')
osclient = OpenSearchClient()
reranker = ReRanker()
intfloat = ReRanker(model_name='intfloat/simlm-msmarco-reranker')


# In[14]:


query = "How did the United States respond to the Soviet Union's advancements in space?"
kw_index = 'impact-theory-minilm-196'
vec_index = 'impact-theory-minilm-196'


# In[92]:


def run_evaluation( dataset: Dict[str, List[str]], 
                    retriever: OpenSearchClient,
                    reranker: ReRanker,
                    kw_index_name: str, 
                    vector_index_name: str,
                    response_size: int=10,
                    top_k: int=5,
                    chunk_size: int=196,
                    rerank_all_responses: bool=False,
                    ) -> Tuple[int, int, int, int]:

    top_k = top_k if top_k else response_size
    reranker_name = reranker.model_name if rerank_all_responses else "None"
    
    results_dict = {'n':response_size, 
                    'top_k': top_k, 
                    'Retriever': retriever.model_name_or_path, 
                    'Ranker': reranker_name,
                    'chunk_size': chunk_size,
                    'kw_recall': 0,
                    'vector_recall': 0,
                    'hybrid_recall':0,
                    'total_questions':0
                    }
    for doc_id, questions in tqdm(dataset.items(), 'Questions'):
        for q in questions:
            results_dict['total_questions'] += 1
            
            #make calls to OpenSearch host of: Keyword, Vector, and Hybrid
            kw_response = retriever.keyword_search(query=q, index=kw_index_name, size=response_size)
            vector_response = retriever.vector_search(query=q, index=vector_index_name, size=response_size)
            hybrid_response = retriever.hybrid_search(q, kw_index_name, vector_index_name, kw_size=response_size, vec_size=response_size)

            #rerank returned responses if rerank_all is True
            if rerank_all_responses:
                kw_response = reranker.rerank(kw_response, q, top_k=top_k)
                vector_response = reranker.rerank(vector_response, q, top_k=top_k)
                hybrid_response = reranker.rerank(hybrid_response, q, top_k=top_k)
                
            #collect doc_ids to check for document matches (include only top_k if top_k > 0)
            kw_doc_ids = [res['_source']['doc_id'] for res in kw_response][:top_k]
            vector_doc_ids = [res['_source']['doc_id'] for res in vector_response][:top_k]
            hybrid_doc_ids = [res['_source']['doc_id'] for res in hybrid_response][:top_k]
            
            #increment recall counters as appropriate
            if doc_id in kw_doc_ids:
                results_dict['kw_recall'] += 1
            if doc_id in vector_doc_ids:
                results_dict['vector_recall'] += 1
            if doc_id in hybrid_doc_ids:
                results_dict['hybrid_recall'] += 1

    #use raw counts to calculate final scores
    calc_recall_scores(results_dict)
    
    return results_dict
        


# In[93]:


def calc_recall_scores(results_dict: Dict[str, Union[str, int]]):
    for prefix in ['kw', 'vector', 'hybrid']:
        results_dict[f'{prefix}_score'] = round(results_dict[f'{prefix}_recall']/results_dict['total_questions'],2)


# In[94]:


def record_results(results_dict: Dict[str, Union[str, int]], dir_outpath: str=None) -> None:
    #write results to output file
    if dir_outpath:
        time_marker = datetime.now().strftime("%Y-%m-%d:%H:%M:%S")
        path = os.path.join(dir_outpath, f'retrieval_eval_{chunk_size}_{time_marker}.json')
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=4)


# In[113]:


chunk_size = 196
all_results = []
for x in range(60,61):
    results = run_evaluation( dataset=dataset, 
                              retriever=osclient, 
                              reranker=intfloat,
                              kw_index_name=kw_index, 
                              vector_index_name=vec_index, 
                              response_size=x, 
                              top_k=10,
                              rerank_all_responses=True,
                            )
    all_results.append(results)
record_results(all_results, dir_outpath='./practice_data/')


# In[121]:


# %%time
# query = 'How do I get ahead in life?'
# resp = osclient.hybrid_search(query, kw_index, vec_index, kw_size=60, vec_size=60)
# intfloat.rerank(resp, query)


# In[ ]:




