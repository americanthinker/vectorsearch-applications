import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Union, Dict, Tuple
from loguru import logger
import tiktoken
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from llama_index.text_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer
from torch import cuda
import pathlib


## Set of helper functions that support data preprocessing 
class FileIO:
    '''
    Convenience class for saving and loading data in various formats to/from disk.
    Currently supports parquet and json formats.
    '''

    def save_as_parquet(self, 
                        file_path: str, 
                        data: Union[List[dict], pd.DataFrame], 
                        overwrite: bool=False) -> None:
        '''
        Saves DataFrame to disk as a parquet file.  Removes the index. 
        '''
        if isinstance(data, list):
           data = self._convert_toDataFrame(data)
        if not file_path.endswith('parquet'):
            file_path = self._rename_file_extension(file_path, 'parquet')
        self._check_file_path(file_path, overwrite=overwrite)
        data.to_parquet(file_path, index=False)
        logger.info(f'DataFrame saved as parquet file here: {file_path}')
        
    def _convert_toDataFrame(self, data: List[dict]) -> pd.DataFrame:
        return pd.DataFrame().from_dict(data)

    def _rename_file_extension(self, file_path: str, extension: str):
        '''
        Renames file with appropriate extension (txt or parquet) if file_path
        does not already have correct extension.
        '''
        prefix = os.path.splitext(file_path)[0]
        file_path = prefix + '.' + extension
        return file_path

    def _check_file_path(self, file_path: str, overwrite: bool) -> None:
        '''
        Checks for existence of file and overwrite permissions.
        '''
        if os.path.exists(file_path) and overwrite == False:
            raise FileExistsError(f'File by name {file_path} already exists, try using another file name or set overwrite to True.')
        elif os.path.exists(file_path):
            os.remove(file_path)
        else: 
            file_name = os.path.basename(file_path)
            dir_structure = file_path.replace(file_name, '')
            pathlib.Path(dir_structure).mkdir(parents=True, exist_ok=True)
    
    def load_parquet(self, file_path: str, verbose: bool=True) -> List[dict]:
        '''
        Loads parquet from disk, converts to pandas DataFrame as intermediate
        step and outputs a list of dicts (docs).
        '''
        df = pd.read_parquet(file_path)
        vector_labels = ['content_vector', 'image_vector', 'content_embedding']
        for label in vector_labels:
            if label in df.columns:
                df[label] = df[label].apply(lambda x: x.tolist())
        if verbose:
            memory_usage = round(df.memory_usage().sum()/(1024*1024),2)
            print(f'Shape of data: {df.values.shape}')
            print(f'Memory Usage: {memory_usage}+ MB')
        list_of_dicts = df.to_dict('records')
        return list_of_dicts
    
    def save_as_json(self, 
                     file_path: str, 
                     data: List[dict], 
                     indent: int=4,
                     overwrite: bool=False
                     ) -> None:
        if not file_path.endswith('json'):
            file_path = self._rename_file_extension(file_path, 'json')
        self._check_file_path(file_path, overwrite=overwrite)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f'Data saved as json file here: {file_path}')

class Utilities: 

    def json_data_loader(self, file_path: str):
        with open(file_path) as f:
            data = json.loads(f.read())
        return data
        
    def create_title(self, file_name: str, 
                    replacements: list=[('_', ' '), ('\s+', ' '), ('Microsoft Word - ', '')]
                    ) -> str:
        '''
        This function was created because several files from the original dataset did 
        not have content-identifying titles. 
        '''
        replacements = replacements
        title = file_name
        for old, new in replacements:
            title = re.sub(old, new, title).strip()
        title = os.path.splitext(title)[0]
        return title

    def create_video_url(self, video_id: str, playlist_id: str):
        '''
        Creates a hyperlink to a video episode given a video_id and playlist_id.
        '''
        return f'https://www.youtube.com/watch?v={video_id}&list={playlist_id}'
        
    def get_content_lengths(self, 
                            list_of_dicts: List[dict], 
                            use_tokens: bool=True, 
                            encoding: str="cl100k_base") -> pd.DataFrame:
        '''
        Given a list of dictionaries with a content field, returns a DataFrame
        of content length split on whitespace. 
        '''
        tokenizer = tiktoken.get_encoding(encoding_name=encoding)
        if use_tokens:
            lens = list(map(lambda x: len(tokenizer.encode(x['content'])), list_of_dicts))
        else:
            lens = list(map(lambda x: len(x['content'].split()), list_of_dicts))
        return pd.DataFrame(lens, columns=['lengths'])
        
    def clean_dict(self, _dict: dict, keys_to_remove: list=['meta', 'split_id']) -> None:
        '''
        Removes keys from dict. 
        '''
        if isinstance(keys_to_remove, str):
            keys_to_remove = [keys_to_remove]
        for key in keys_to_remove:
            try:
                del _dict[key]
            except KeyError:
                continue

class Splitters:

    def split_corpus(self,
                     corpus: List[dict], 
                     text_splitter: SentenceSplitter, 
                     create_dict: bool=True
                     ) -> List[dict]:
        text_chunks = []
        doc_idxs = []
        for i, doc in enumerate(tqdm(corpus, 'Docs')):
            splits = text_splitter.split_text(doc.get('content', ''))
            text_chunks.extend(splits)
            doc_idxs.extend([i] * len(splits))
        if create_dict: 
            split_dict = defaultdict(list)
            data = zip(doc_idxs, text_chunks)
            for i, chunk in data:
                split_dict[i].append(chunk)
            return split_dict
        return doc_idxs, text_chunks
        
    def easy_split_corpus( self,
                  corpus: List[dict], 
                  text_splitter: SentenceSplitter, 
                  content_key: str='content'
                  ) -> List[dict]:
        '''
        Given a corpus of "documents" with text content, this function splits the 
        content field into chunks sizes as specified by the text_splitter param. 
        '''
        text_chunks = {}
        for doc in tqdm(corpus, 'Docs'):
            video_id = doc['video_id']
            splits = text_splitter.split_text(doc.get(content_key, ''))
            text_chunks[video_id] = splits
        return text_chunks
    
class Vectorizor:

    def __init__(self, model_name_or_path: str='all-MiniLM-L6-v2'):
        self.model_type = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path)

    def add_vector(self, 
                   doc: dict, 
                   content_field: str='content',
                   device: str='cuda:0' if cuda.is_available() else 'cpu'
                   ) -> None:
        chunk = doc[content_field]
        vector = self.model.encode(sentences=chunk,
                                   show_progress_bar=False,
                                   device=device)
        doc['vector'] = vector.tolist()
        return doc
    
    def add_vector_batch(self, 
                        docs: List[dict], 
                        content_field: str='content',
                        device: str='cuda:0' if cuda.is_available() else 'cpu'
                        ) -> None:
        sentence_chunks = [d[content_field] for d in docs]
        vectors = self.model.encode(sentences=sentence_chunks,
                                    show_progress_bar=True,
                                    device=device)
        for i, d in enumerate(docs):
            d['vector'] = vectors[i].tolist()
         
        return docs

    def encode_from_dict(self, split_dict: dict, device: str='cuda:0') -> Dict[int, Tuple[str, np.array]]:
        '''
        Encode text to vectors from a dictionary where digits are keys, 
        with each key representing an entire document (podcast)
        '''
        merged = defaultdict(list)
        for key in tqdm(split_dict.keys(), "Docs"):
            chunks = split_dict[key]
            vectors = self.model.encode(sentences=chunks, show_progress_bar=False, device=device)
            merged[key] = list(zip(chunks, vectors))
        return merged

    def join_metadata(self, corpus: List[dict], merged_dict: dict, create_doc_id: bool=True) -> List[dict]:
        '''
        For each text-chunk/vector pair in merged_dict, create a single 
        dictionary that joins text, vector, and podcast metadata from the
        original corpus of documents
        '''
        joined_documents = []
        for index in merged_dict:
            meta = corpus[index]
            for i, _tuple in enumerate(merged_dict[index]):
                doc = {k:v for k,v in meta.items() if k != 'content'}
                doc['content'] = _tuple[0]
                doc['content_embedding'] = _tuple[1].tolist()
                if create_doc_id:
                    if doc.get('video_id'):
                        doc['doc_id'] = doc['video_id'] + '_' + str(i)
                joined_documents.append(doc)
        return joined_documents
    

