import time
import os
from typing import List, Tuple
from itertools import groupby

#external libraries
from rich import print
from torch import cuda
from numpy import ndarray
from tqdm.notebook import tqdm

#external files
from src.preprocessor.preprocessing import FileIO # bad ass tokenizer library for use with OpenAI LLMs 
from llama_index.text_splitter import SentenceSplitter #one of the best on the market
from sentence_transformers import SentenceTransformer

def chunk_data(data: List[dict], text_splitter: SentenceSplitter, content_field='content') -> List[List[str]]:
    return [text_splitter.split_text(d[content_field]) for d in tqdm(data, 'CHUNKING')]

def create_vectors(content_splits: List[List[str]], 
                   model: SentenceTransformer, 
                   device: str
                   ) -> list[tuple[str, ndarray[float]]]:
    text_vector_tuples = []
    for chunk in tqdm(content_splits, 'VECTORS'):
        vectors = model.encode(chunk, show_progress_bar=False, device=device)
        text_vector_tuples.append(list(zip(chunk, vectors)))
    return text_vector_tuples

def join_docs(corpus: List[dict], 
              tuples: List[Tuple[str, float]],
              unique_id_field: str='video_id',
              content_field: str='content',
              embedding_field: str='content_embedding'
              ) -> List[dict]:
    docs = []
    for i, d in enumerate(corpus):
        for j, episode in enumerate(tuples[i]):
            doc = {k:v for k,v in d.items() if k != content_field}
            unique_id = doc[unique_id_field]
            doc['doc_id'] = f'{unique_id}_{j}'
            doc[content_field] = episode[0]
            doc[embedding_field] = episode[1].tolist()
            docs.append(doc)
    return docs
    
def convert_raw_data(raw_data: list[dict]) -> list[dict]:
    '''
    Converts raw YouTube json to correct format for 
    indexing on Weaviate. i.e. drops unused fields, 
    and coerces data types. 
    '''
    drops = ['channelId', 'isOwnerViewing', 'isCrawlable', 'allowRatings', \
             'author', 'isPrivate', 'isUnpluggedCorpus', 'isLiveContent']
    data = list(raw_data.values())
    for d in data:
        d['thumbnail_url'] = d['thumbnail']['thumbnails'][1].get('url')
        d['lengthSeconds'] = int(d['lengthSeconds'])
        d['viewCount'] = int(d['viewCount'])
        del d['thumbnail']
        for field in drops:
            del d[field]
    return data
    
def create_dataset(corpus: List[dict],
                   embedding_model: SentenceTransformer,
                   text_splitter: SentenceSplitter,
                   file_outpath_prefix: str='./huberman-labs-minilm',
                   unique_id_field: str='video_id',
                   content_field: str='content',
                   embedding_field: str='content_embedding',
                   overwrite_existing: bool=False,
                   device: str='cuda:0' if cuda.is_available() else 'cpu'
                   ) -> list[dict]:
    '''
    Given a raw corpus of data, this function creates a new dataset where each dataset 
    doc contains episode metadata and it's associated text chunk and vector representation. 
    Output is directly saved to disk. 
    '''
    
    io = FileIO()
    chunk_size = text_splitter.chunk_size
    file_path = f'{file_outpath_prefix}-{chunk_size}.parquet'
    # fail early prior to kicking off expensive job
    if os.path.exists(file_path) and overwrite_existing == False:
            raise FileExistsError(f'File by name {file_path} already exists, try using another file name or set overwrite_existing to True.')
    print(f'Creating dataset using chunk_size: {chunk_size}')

    start = time.perf_counter()

    content_splits = chunk_data(corpus, text_splitter, content_field)
    text_vector_tuples = create_vectors(content_splits, embedding_model, device)
    try:
        joined_docs = join_docs(corpus, text_vector_tuples, unique_id_field, content_field, embedding_field)
    except Exception as e:
        print(f'Failed to join docs due to: {e}. Try manually joining docs using the returned text_vector_tuples and your original corpus')
        return text_vector_tuples
    try:
        io.save_as_parquet(file_path=file_path, data=joined_docs, overwrite=overwrite_existing)
    except Exception as e:
        print(f'Dataset not saved to disk as parquet file due to: {e}')
    end = time.perf_counter() - start
    print(f'Total Time to process dataset of chunk_size ({chunk_size}): {round(end/60, 2)} minutes')
    return joined_docs

def groupby_episode(data: List[dict], key_field: str='video_id') -> List[List[dict]]:
    '''
    Separates entire Impact Theory corpus into individual 
    lists of discrete episodes.
    '''
    episodes = []
    for key, group in groupby(data, lambda x: x[key_field]):
        episode = [chunk for chunk in group]
        episodes.append(episode)
    return episodes

def create_parent_chunks(episode_list: List[list], window_size: int=2) -> List[dict]:
    '''
    Creates parent chunks from original chunk of text, for use with 
    small to big retrieval.  Window size sets number of chunks before
    and after the original chunk.  For example a window_size of 2 will 
    return five joined chunks.  2 chunks before original, the original, 
    and 2 chunks after the original.  Chunks are kept in sequence by 
    using the doc_id field. 
    '''
    parent_chunks = []
    for episode in episode_list:
        contents = [d['content'] for d in episode]
        for i, d in enumerate(episode):
            doc_id = d['doc_id']
            start = max(0, i-window_size)
            end = i+window_size+1
            chunk = ' '.join(contents[start:end])
            parent_chunks.append({doc_id:chunk})
    return parent_chunks

def create_parent_chunk_cache(parent_chunks: List[dict]) -> dict:
    '''
    Creates a simple in-memory cache for quick parent chunk lookup.
    Used for small-to-big retrieval in a RAG system.
    '''
    content_cache = {}
    for chunk in parent_chunks:
        for k,v in chunk.items():
            content_cache[k] = v
    return content_cache