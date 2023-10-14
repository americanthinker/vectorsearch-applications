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
    
    def sentence_splitter(self, text: str) -> List[str]:
        '''
        Given a piece of text, returns text split into a list of sentences at 
        defined sentence breaks. 

        Args
        -----
        text : str
            Piece of text or document in string format. 

        Returns
        --------
            List of sentences demarcated by sentence boundaries (period, ? !).

        '''
        #Adding a period at the end of the string to account for text (such as found in Powerpoint presentations)
        #that do not end with sentence boundaries i.e. periods, question marks, etc.
        text = text + '.'
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|Col|Gen|Pfc|Spc|Maj|Lt|Adm|Capt)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov)"
        digits = "([0-9])"
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
        text = re.sub(digits + "[.]" + " " + digits, "\\1<prd>\\2", text)
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D","Ph<prd>D<prd>")
        if "e.g." in text: text = text.replace("e.g.","e<prd>g<prd>")
        if "e. g." in text: text = text.replace("e. g.","e<prd>g<prd>")
        if "i.e." in text: text = text.replace("i.e.","i<prd>e<prd>") 
        if "i. e." in text: text = text.replace("i. e.","i<prd>e<prd>") 
        if "et. al." in text: text = text.replace("et. al.", 'et<prd>al<prd>')
        if "Fig. " in text: text = text.replace("Fig. ", "Fig.<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        sentences = [re.sub('\s+', ' ', sent) for sent in sentences]

        #remove the period at the end of the last sentence, that was inserted at the beginning of this function
        if sentences:
            last_sentence = sentences[-1][:-1]
            sentences = sentences[:-1]
            if last_sentence:
                sentences.append(last_sentence)

        return sentences

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