from weaviate import Client, AuthApiKey
from sentence_transformers import SentenceTransformer
from typing import List, Union
from torch import cuda

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
        self.fields = ['title', 'video_id', 'length', 'thumbnail_url', 'views', 'episode_url', 'doc_id', 'content']  # 'playlist_id', 'channel_id', 'author'
        
    def show_classes(self):
        '''
        Shows all available classes (indexes) on the Weaviate instance.
        '''
        classes = self.cluster.get_nodes_status()[0]['shards']
        if classes:
            return [d['class'] for d in classes]
        else: 
            return "No classes found on cluster."

    def show_class_info(self):
        '''
        Shows all information related to the classes (indexes) on the Weaviate instance.
        '''
        classes = self.cluster.get_nodes_status()[0]['shards']
        if classes:
            return [d for d in classes]
        else: 
            return "No classes found on cluster."

    def _format_response(self, 
                         response: dict,
                         class_name: str
                         ) -> List[dict]:
        '''
        Formats json response from Weaviate into a list of dictionaries.
        Expands _additional fields if present into primary dictionary.
        '''
        if response.get('errors'):
            return response['errors'][0]['message']
        results = []
        hits = response['data']['Get'][class_name]
        for d in hits:
            temp = {k:v for k,v in d.items() if k != '_additional'}
            if d.get('_additional'):
                for key in d['_additional']:
                    temp[key] = d['_additional'][key]
            results.append(temp)
        return results
        
    def keyword_search(self,
                       query: str,
                       class_name: str,
                       properties: List[str]=['content'],
                       limit: int=10,
                       return_raw: bool=False) -> Union[dict, List[dict]]:
        '''
        Executes Keyword (BM25) search. 

        Args
        ----
        query: str
            User query.

        class_name: str
            Class (index) to search.
        
        properties: List[str]
            List of properties to search across.
        
        limit: int=10
            Number of results to return.
        
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        response = (self.query
                    .get(class_name,self.fields)
                    .with_bm25(query=query, properties=properties)
                    .with_additional(['score', "id"])
                    .with_limit(limit)
                    .do()
                    )
        if return_raw:
            return response
        else: 
            return self._format_response(response, class_name)

    def vector_search(self,
                      query: str,
                      class_name: str,
                      limit: int=10,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                     ) -> Union[dict, List[dict]]:
        '''
        Executes vector search using embedding model defined on instantiation 
        of WeaviateClient instance.
        
        Args
        ----
        query: str
            User query.

        class_name: str
            Class (index) to search.
        
        limit: int=10
            Number of results to return.
        
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        query_vector = self.model.encode(query, device=device).tolist()
        response = (
                    self.query
                    .get(class_name, self.fields)
                    .with_near_vector({"vector": query_vector})
                    .with_limit(limit)
                    .with_additional(['distance'])
                    .do()
                    )
        if return_raw:
            return response
        else: 
            return self._format_response(response, class_name)

    def hybrid_search(self,
                      query: str,
                      class_name: str,
                      properties: List[str]=['content'],
                      alpha: float=0.5,
                      limit: int=10,
                      where_filter: dict=None,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                     ) -> Union[dict, List[dict]]:
        '''
        Executes Hybrid (BM25 + Vector) search.
        
        Args
        ----
        query: str
            User query.

        class_name: str
            Class (index) to search.
        
        properties: List[str]
            List of properties to search across.
        
        alpha: float=0.5
            Weighting factor for BM25 and Vector search.
            alpha can be any number from 0 to 1, defaulting to 0.5:
                alpha = 0 executes a pure keyword search method (BM25)
                alpha = 0.5 weighs the BM25 and vector methods evenly
                alpha = 1 executes a pure vector search method

        limit: int=10
            Number of results to return.
        
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        query_vector = self.model.encode(query, device=device).tolist()
        response = (
                    self.query
                    .get(class_name, self.fields)
                    .with_hybrid(query=query,
                                 alpha=alpha,
                                 vector=query_vector,
                                 properties=properties,
                                 fusion_type='relativeScoreFusion') #hard coded option for now
                    .with_additional(["score", "explainScore"])
                    .with_limit(limit)
                    )
        
        response = response.with_where(where_filter).do() if where_filter else response.do()
        if return_raw:
            return response
        else: 
            return self._format_response(response, class_name)