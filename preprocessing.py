import os
import re
import json
import pandas as pd
from typing import List
from loguru import logger
import tiktoken
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import cuda


## Set of helper functions that support data preprocessing 

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

    def get_content_lengths(self, list_of_dicts: List[dict], 
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

        # @classmethod
        # def split_data(cls, 
        #                data: List[dict], 
        #                split_length: int, 
        #                semantic_splitting: bool=False) -> List[dict]:
        #     '''
        #     Helper function primarily desinged to avoid making another pass through the 
        #     data/corpus to create a data list in smaller or larger chunk sizes new indexes.  
        #     '''
        #     fresh_data = deepcopy(data)
        #     docprocessor = DocumentProcessor('fake_folder', split_length=split_length)
        #     data_splits = []
        #     progress = tqdm(unit=": Splitting Data", total=len(data))
        #     with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor: 
        #         futures = [executor.submit(docprocessor.split, d, split_length) for d in fresh_data]
        #         for future in as_completed(futures):
        #             data_splits.append(future.result())
        #             progress.update(1)

        #     # #remove None data_splits 
        #     data_splits = [item for item in data_splits if item]
        #     #flatten list of dicts
        #     data_splits = [d for group in data_splits for d in group if len(d['content'].split()) >= 2 if semantic_splitting]
        #     return data_splits

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