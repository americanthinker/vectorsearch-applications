from weaviate.auth import AuthApiKey
from weaviate.collections.classes.internal import (MetadataReturn, QueryReturn,
                                                   MetadataQuery)
import weaviate
from weaviate.classes.config import Property
from weaviate.config import ConnectionConfig
from dataclasses import dataclass
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Callable, Union
from torch import cuda
from tqdm import tqdm
import time

class WeaviateWCS:
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
        Will also support OpenAI text-embedding-ada-002 model.  This param enables 
        the use of most leading models on MTEB Leaderboard: 
        https://huggingface.co/spaces/mteb/leaderboard
    openai_api_key: str=None
        The API key for the OpenAI API. Only required if using OpenAI text-embedding-ada-002 model.
    '''    
    def __init__(self, 
                 endpoint: str=None,
                 api_key: str=None,
                 model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2',
                 embedded: bool=False,
                 openai_api_key: str=None,
                 **kwargs
                ):
        if embedded:
            self._client = weaviate.connect_to_embedded(**kwargs)
        else: 
            auth_config = AuthApiKey(api_key=api_key) 
            self._client = weaviate.connect_to_wcs(cluster_url=endpoint, auth_credentials=auth_config, skip_init_checks=True)   
        self._model_name_or_path = model_name_or_path
        self._openai_model = False
        if self._model_name_or_path == 'text-embedding-ada-002':
            if not openai_api_key:
                raise ValueError(f'OpenAI API key must be provided to use this model: {self._model_name_or_path}')
            self.model = OpenAI(api_key=openai_api_key)
            self._openai_model = True
        else: 
            self.model = SentenceTransformer(self._model_name_or_path) if self._model_name_or_path else None

        self.return_properties = ['title', 'videoId', 'content']  # 'playlist_id', 'channel_id', 'author'

    def _connect(self):
        '''
        Connects to Weaviate instance.
        '''
        if not self._client.is_connected():
            self._client.connect()

    def show_all_collections(self, 
                             detailed: bool=False,
                             max_details: bool=False
                             ) -> list[str] | dict:
        '''
        Shows all available collections(indexes) on the Weaviate cluster.
        By default will only return list of collection names.
        Otherwise, increasing details about each collection can be returned.
        '''
        self._connect()
        collections = self._client.collections.list_all(simple=not max_details)
        self._client.close()
        if not detailed and not max_details:
            return list(collections.keys())
        else:
            if not any(collections):
                print('No collections found on host')
            return collections

    def show_collection_config(self, collection_name: str) -> ConnectionConfig:
        '''
        Shows all information of a specific collection. 
        '''
        self._connect()
        if self._client.collections.exists(collection_name):
            collection = self.show_all_collections(max_details=True)[collection_name]
            self._client.close()
            return collection
        else: 
            print(f'Collection "{collection_name}" not found on host')

    def show_collection_properties(self, collection_name: str) -> Union[dict, str]:
        '''
        Shows all properties of a collection (index) on the Weaviate instance.
        '''
        self._connect()
        if self._client.collections.exists(collection_name):
            collection = self.show_all_collections(max_details=True)[collection_name]
            self._client.close()
            return collection.properties
        else: 
            print(f'Collection "{collection_name}" not found on host')
    
    def delete_collection(self, collection_name: str) -> str:
        '''
        Deletes a collection (index) on the Weaviate instance, if it exists.
        '''
        self._connect()
        if self._client.collections.exists(collection_name):
            try:
                self._client.collections.delete(collection_name)
                self._client.close()
                print(f'Collection "{collection_name}" deleted')
            except Exception as e:
                print(f'Error deleting collection, due to: {e}')
        else: 
            print(f'Collection "{collection_name}" not found on host')
        
    def format_response(self, 
                        response: QueryReturn,
                        ) -> list[dict]:
        '''
        Formats json response from Weaviate into a list of dictionaries.
        Expands _additional fields if present into top-level dictionary.
        '''
        results = [{**d.properties, **self._get_meta(d.metadata)} for d in response.objects]
        return results

    def _get_meta(self, metadata: MetadataReturn):
        '''
        Extracts metadata from MetadataQuery object if meta exists.
        '''
        temp_dict = metadata.__dict__
        return {k:v for k,v in temp_dict.items() if v}

    # def update_ef_value(self, class_name: str, ef_value: int) -> str:
    #     '''
    #     Updates ef_value for a collection (index) on the Weaviate instance.
    #     '''
    #     self.schema.update_config(class_name=class_name, config={'vectorIndexConfig': {'ef': ef_value}})
    #     print(f'ef_value updated to {ef_value} for collection {class_name}')
    #     return self.show_class_config(class_name)['vectorIndexConfig']
        
    def keyword_search(self,
                       request: str,
                       collection_name: str,
                       query_properties: list[str]=['content'],
                       limit: int=10,
                       where_filter: dict=None,
                       return_properties: list[str]=None,
                       return_raw: bool=False) -> Union[dict, list[dict]]:
        '''
        Executes Keyword (BM25) search. 

        Args
        ----
        request: str
            User query.
        collection_name: str
            Collection (index) to search.
        query_properties: list[str]
            list of properties to search across.
        limit: int=10
            Number of results to return.
        where_filter: dict=None
            Property filter to apply to search results.
        return_properties: list[str]=None
            list of properties to return in response.
            If None, returns self.return_properties.
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        self._connect()
        return_properties = return_properties if return_properties else self.return_properties
        connection = self._client.collections.get(collection_name)
        response = connection.query.bm25(query=request,
                                         query_properties=query_properties,
                                         limit=limit,
                                         return_metadata=MetadataQuery(score=True),
                                         return_properties=return_properties)
        # response = response.with_where(where_filter).do() if where_filter else response.do()
        self._client.close()
        if return_raw:
            return response
        else: 
            return self.format_response(response)

    def vector_search(self,
                      request: str,
                      collection_name: str,
                      limit: int=10,
                      return_properties: list[str]=None,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                      ) -> dict | list[dict]:
        '''
        Executes vector search using embedding model defined on instantiation 
        of WeaviateClient instance.
        
        Args
        ----
        request: str
            User query.
        collection_name: str
            Collection (index) to search.
        limit: int=10
            Number of results to return.
        return_properties: list[str]=None
            list of properties to return in response.
            If None, returns all properties.
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        device: str
            Device to use for encoding query.
        '''
        self._connect()
        return_properties = return_properties if return_properties else self.return_properties
        query_vector = self._create_query_vector(request, device=device)
        connection = self._client.collections.get(collection_name)
        response = connection.query.near_vector(near_vector=query_vector,
                                                limit=limit,
                                                return_metadata=MetadataQuery(distance=True, 
                                                                                explain_score=True,
                                                                                ),
                                                return_properties=return_properties)
        #  response = response.with_where(where_filter).do() if where_filter else response.do()
        self._client.close()
        if return_raw:
            return response
        else: 
            return self.format_response(response)    
    
    def _create_query_vector(self, query: str, device: str) -> list[float]:
        '''
        Creates embedding vector from text query.
        '''
        return self.get_openai_embedding(query) if self._openai_model else self.model.encode(query, device=device).tolist()
    
    def get_openai_embedding(self, query: str) -> list[float]:
        '''
        Gets embedding from OpenAI API for query.
        '''
        embedding = self.model.embeddings.create(input=query, model='text-embedding-ada-002').model_dump()
        if embedding:
            return embedding['data'][0]['embedding']
        else:
           raise ValueError(f'No embedding found for query: {query}')
        
    def hybrid_search(self,
                      request: str,
                      collection_name: str,
                      query_properties: list[str]=['content'],
                      alpha: float=0.5,
                      limit: int=10,
                      where_filter: dict=None,
                      return_properties: list[str]=None,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                     ) -> Union[dict, list[dict]]:
        '''
        Executes Hybrid (BM25 + Vector) search.
        
        Args
        ----
        request: str
            User query.
        collection_name: str
            Collection (index) to search.
        query_properties: list[str]
            list of properties to search across (using BM25)
        alpha: float=0.5
            Weighting factor for BM25 and Vector search.
            alpha can be any number from 0 to 1, defaulting to 0.5:
                alpha = 0 executes a pure keyword search method (BM25)
                alpha = 0.5 weighs the BM25 and vector methods evenly
                alpha = 1 executes a pure vector search method
        limit: int=10
            Number of results to return.
        where_filter: dict=None
            Property filter to apply to search results.
        return_properties: list[str]=None
            list of properties to return in response.
            If None, returns all properties.
        return_raw: bool=False
            If True, returns raw response from Weaviate.
        '''
        self._connect()
        return_properties = return_properties if return_properties else self.return_properties
        query_vector = self._create_query_vector(request, device=device)
        connection = self._client.collections.get(collection_name)
        response = connection.query.hybrid(query=request,
                                           query_properties=query_properties,
                                           vector=query_vector,
                                           alpha=alpha,
                                           limit=limit,
                                           return_metadata=MetadataQuery(score=True, distance=True),
                                           return_properties=return_properties)
        # response = response.with_where(where_filter).do() if where_filter else response.do()
        self._client.close()
        if return_raw:
            return response
        else: 
            return self.format_response(response)
        
        
class WeaviateIndexer:

    def __init__(self,
                 client: WeaviateWCS
                 ):
        '''
        Class designed to batch index documents into Weaviate. Instantiating
        this class will automatically configure the Weaviate batch client.
        '''
    
        self._client = client._client

    def _connect(self):
        '''
        Connects to Weaviate instance.
        '''
        if not self._client.is_connected():
            self._client.connect()

    def create_collection(self, 
                          collection_name: str, 
                          properties: list[Property],
                          description: str=None,
                          **kwargs
                          ) -> str:
        '''
        Creates a collection (index) on the Weaviate instance.
        '''
        if collection_name.find('-') != -1:
            raise ValueError('Collection name cannot contain hyphens')
        try:
            self._connect()
            collection = self._client.collections.create(
                                                        name=collection_name,
                                                        description=description,
                                                        properties=properties,
                                                        **kwargs
                                                        )
            if self._client.collections.exists(collection_name):
                print(f'Collection "{collection_name}" created')
            else:
                print(f'Collection not found at the moment, try again later')
            self._client.close()
        except Exception as e:
            print(f'Error creating collection, due to: {e}')

    def batch_index_data(self,
                         data: list[dict], 
                         collection_name: str,
                         vector_property: str='content_embedding', 
                         return_batch_errors: bool=True,
                         properties: list[Property]=None,
                         description: str=None,
                         **kwargs
                         ) -> None:
        '''
        Batch function for fast indexing of data onto Weaviate cluster. 
        This method assumes that self._client.batch is already configured.
        '''
        self._connect()
        if not self._client.collections.exists(collection_name):
            print(f'Collection "{collection_name}" not found on host, creating Collection first...')
            if properties is None:
                raise ValueError(f'Tried to create Collection <{collection_name}> but no properties were provided.')
            self.create_collection(collection_name=collection_name, 
                                   properties=properties,
                                   description=description,
                                   **kwargs)
            self._client.close()
        start = time.perf_counter()
        self._connect()
        collection = self._client.collections.get(collection_name)
        with collection.batch.dynamic() as batch:
            for doc in tqdm(data):
                try:
                    batch.add_object(properties={k:v for k,v in doc.items() if k != vector_property},
                                        vector=doc[vector_property])
                except Exception as e:
                    print(e)
                    continue
        end = time.perf_counter() - start
        self._client.close()
        print(f'Batch job completed in {round(end/60, 2)} minutes.')
        if return_batch_errors:
            return {'batch_errors':batch.number_errors, 
                    'failed_objects':self._client.batch.failed_objects,
                    'failed_references': self._client.batch.failed_references}
            

# @dataclass
# class WhereFilter:

#     '''
#     Simplified interface for constructing a WhereFilter object.

#     Args
#     ----
#     path: list[str]
#         list of properties to filter on.
#     operator: str
#         Operator to use for filtering. Options: ['And', 'Or', 'Equal', 'NotEqual', 
#         'GreaterThan', 'GreaterThanEqual', 'LessThan', 'LessThanEqual', 'Like', 
#         'WithinGeoRange', 'IsNull', 'ContainsAny', 'ContainsAll']
#     value[dataType]: Union[int, bool, str, float, datetime]
#         Value to filter on. The dataType suffix must match the data type of the 
#         property being filtered on. At least and only one value type must be provided. 
#     '''
#     path: list[str]
#     operator: str
#     valueInt: int=None
#     valueBoolean: bool=None
#     valueText: str=None
#     valueNumber: float=None
#     valueDate = None

#     def post_init(self):
#         operators = ['And', 'Or', 'Equal', 'NotEqual','GreaterThan', 'GreaterThanEqual', 'LessThan',\
#                       'LessThanEqual', 'Like', 'WithinGeoRange', 'IsNull', 'ContainsAny', 'ContainsAll']
#         if self.operator not in operators:
#             raise ValueError(f'operator must be one of: {operators}, got {self.operator}')
#         values = [self.valueInt, self.valueBoolean, self.valueText, self.valueNumber, self.valueDate]
#         if not any(values):
#             raise ValueError('At least one value must be provided.')
#         if len(values) > 1:
#             raise ValueError('At most one value can be provided.')
    
#     def todict(self):
#         return {k:v for k,v in self.__dict__.items() if v is not None}