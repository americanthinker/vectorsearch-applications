from weaviate import Client, AuthApiKey
from sentence_transformers import SentenceTransformer
from typing import List, Union


class WeaviateClient(Client):
    '''
    A python native Weaviate Client class that encapsulates Weaviate functionalities 
    in one object. Several convenience methods are added for ease of use.

    Args
    ----
    api_key: str
        The API key for the Weaviate Cloud Service (WCS) instance.
        https://console.weaviate.cloud/dashboard

    endpoint: str
        The url endpoint for the Weaviate Cloud Service instance.

    model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2'
        The name or path of the SentenceTransformer model to use for vector search.
        Models are hard-coded as SentenceTransformers only for now (works for most 
        leading models on MTEB Leaderboard): https://huggingface.co/spaces/mteb/leaderboard
    '''    
    def __init__(self, 
                 api_key: str,
                 endpoint: str,
                 model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2',
                 **kwargs
                ):
        auth_config = AuthApiKey(api_key=api_key)
        super().__init__(auth_client_secret=auth_config,
                         url=endpoint,
                         **kwargs)    
        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(self.model_name_or_path) if self.model_name_or_path else None
        self.fields = ["title", "content", "docid"]
        
    def show_classes(self):
        '''
        Shows all available classes (indexes) on the Weaviate instance.
        '''
        return [d['class'] for d in self.cluster.get_nodes_status()[0]['shards']]

    def show_class_info(self):
        '''
        Shows all information related to the classes (indexes) on the Weaviate instance.
        '''
        return [d for d in self.cluster.get_nodes_status()[0]['shards']]

    def _format_response(self, 
                         response: dict,
                         class_: str
                         ) -> List[dict]:
        '''
        Formats json response from Weaviate into a list of dictionaries.
        Expands _additional fields if present into primary dictionary.
        '''
        results = []
        hits = response['data']['Get'][class_]
        for d in hits:
            temp = {k:v for k,v in d.items() if k != '_additional'}
            if d.get('_additional'):
                for key in d['_additional']:
                    temp[key] = d['_additional'][key]
            results.append(temp)
        return results
        
    def keyword_search(self,
                       query: str,
                       class_: str,
                       properties: List[str]=['content'],
                       limit: int=10,
                       return_raw: bool=False) -> Union[dict, List[dict]]:
        '''
        Executes Keyword (BM25) search. 

        Args
        ----
        query: str
            User query.

        class_: str
            Class (index) to search.
        
        properties: List[str]
            List of properties to search across.
        
        limit: int=10
            Number of results to return.
        
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        response = (self.query
                    .get(class_,self.fields)
                    .with_bm25(query=query, properties=properties)
                    .with_additional(['score', "id"])
                    .with_limit(limit)
                    .do()
                    )
        if return_raw:
            return response
        else: 
            return self._format_response(response, class_)

    def vector_search(self,
                      query: str,
                      class_: str,
                      properties: List[str]=['content'],
                      limit: int=10,
                      return_raw: bool=False
                     ) -> Union[dict, List[dict]]:
        '''
        Executes Hybrid (BM25 + Vector) search.
        
        Args
        ----
        query: str
            User query.

        class_: str
            Class (index) to search.
        
        properties: List[str]
            List of properties to search across.
        
        limit: int=10
            Number of results to return.
        
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        pass

    def hybrid_search(self,
                      query: str,
                      class_: str,
                      properties: List[str]=['content'],
                      limit: int=10,
                      return_raw: bool=False
                     ) -> Union[dict, List[dict]]:
        '''
        Executes Hybrid (BM25 + Vector) search.
        
        Args
        ----
        query: str
            User query.

        class_: str
            Class (index) to search.
        
        properties: List[str]
            List of properties to search across.
        
        limit: int=10
            Number of results to return.
        
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        pass