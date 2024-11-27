from weaviate.classes.init import Auth
from weaviate.collections.classes.internal import (MetadataReturn, QueryReturn,
                                                   MetadataQuery)
import weaviate
from weaviate.classes.config import Property
from weaviate.classes.query import Filter
from weaviate.config import ConnectionConfig
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from torch import cuda
from tqdm import tqdm
import time
from dataclasses import dataclass

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
        the use of most open source models on MTEB Leaderboard: 
        https://huggingface.co/spaces/mteb/leaderboard

    embedded: bool=False
        If True, connects to an embedded Weaviate instance.

    openai_api_key: str=None
        The API key for the OpenAI API. Only required if using OpenAI text-embedding-ada-002 model.
    '''    

    OPENAI_EMBEDDING_MODELS = ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']

    def __init__(self, 
                 endpoint: str=None,
                 api_key: str=None,
                 model_name_or_path: str='sentence-transformers/all-MiniLM-L6-v2',
                 embedded: bool=False,
                 openai_api_key: str=None,
                 **kwargs
                ):

        self.endpoint = endpoint
        if embedded:
            self._client = weaviate.connect_to_embedded(**kwargs)
        else: 
            self._client = weaviate.connect_to_wcs(cluster_url=endpoint, 
                                                   auth_credentials=Auth.api_key(api_key),
                                                   skip_init_checks=True) 
        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path in self.OPENAI_EMBEDDING_MODELS:
            if not openai_api_key:
                raise ValueError(f'OpenAI API key must be provided to use this model: {self.model_name_or_path}')
            self.model = OpenAI(api_key=openai_api_key)
        else: 
            self.model = SentenceTransformer(self.model_name_or_path) if self.model_name_or_path else None

        self.return_properties = ['guest', 'title', 'summary', 'content', 'video_id', 'doc_id', 'episode_url', 'thumbnail_url']

    def _connect(self) -> None:
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
                          ) -> None:
        '''
        Creates a collection (index) on the Weaviate instance.
        
        Args
        ----
        collection_name: str
            Name of the collection to create.
        properties: list[Property]
            List of properties to add to data objects in the collection.
        description: str=None
            User-defined description of the collection.
        '''
        
        self._connect()
        if self._client.collections.exists(collection_name):
            print(f'Collection "{collection_name}" already exists')
            return 
        else:
            try:
                self._client.collections.create(name=collection_name, 
                                                properties=properties,
                                                description=description,
                                                **kwargs)
                print(f'Collection "{collection_name}" created')
            except Exception as e:
                print(f'Error creating collection, due to: {e}')
        self._client.close()
        return

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

    def show_collection_properties(self, collection_name: str) -> dict | str:
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
    
    def get_doc_count(self, collection_name: str) -> str:
        '''
        Returns the number of documents in a collection.
        '''
        self._connect()
        if self._client.collections.exists(collection_name):
            collection = self._client.collections.get(collection_name)
            aggregate = collection.aggregate.over_all()
            total_count = aggregate.total_count
            print(f'Found {total_count} documents in collection "{collection_name}"')
            return total_count
        else:
            print(f'Collection "{collection_name}" not found on host')
            
    def format_response(self, 
                        response: QueryReturn,
                        ) -> list[dict]:
        '''
        Formats json response from Weaviate into a list of dictionaries.
        Expands _additional fields if present into top-level dictionary.
        '''
        results = [{**o.properties, **self._get_meta(o.metadata)} for o in response.objects]
        return results

    def _get_meta(self, metadata: MetadataReturn):
        '''
        Extracts metadata from MetadataQuery object if meta exists.
        '''
        temp_dict = metadata.__dict__
        return {k:v for k,v in temp_dict.items() if v}
        
    def keyword_search(self,
                       request: str,
                       collection_name: str,
                       query_properties: list[str]=['content'],
                       limit: int=10,
                       filter: Filter=None,
                       return_properties: list[str]=None,
                       return_raw: bool=False
                       ) -> dict | list[dict]:
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
        collection = self._client.collections.get(collection_name)
        response = collection.query.bm25(query=request,
                                         query_properties=query_properties,
                                         limit=limit,
                                         filters=filter,
                                         return_metadata=MetadataQuery(score=True),
                                         return_properties=return_properties)
        # response = response.with_where(where_filter).do() if where_filter else response.do()
        if return_raw:
            return response
        else: 
            return self.format_response(response)

    def vector_search(self,
                      request: str,
                      collection_name: str,
                      limit: int=10,
                      return_properties: list[str]=None,
                      filter: Filter=None,
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
        collection = self._client.collections.get(collection_name)
        response = collection.query.near_vector(near_vector=query_vector,
                                                limit=limit,
                                                filters=filter,
                                                return_metadata=MetadataQuery(distance=True),                                                               
                                                return_properties=return_properties)
        if return_raw:
            return response
        else: 
            return self.format_response(response)    
    
    def _create_query_vector(self, query: str, device: str) -> list[float]:
        '''
        Creates embedding vector from text query.
        '''
        if self.model_name_or_path in self.OPENAI_EMBEDDING_MODELS:
            return self.get_openai_embedding(query)  
        else:
            return self.model.encode(query, device=device).tolist()
    
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
                      filter: Filter=None,
                      return_properties: list[str]=None,
                      return_raw: bool=False,
                      device: str='cuda:0' if cuda.is_available() else 'cpu'
                     ) -> dict | list[dict]:
        '''
        Executes Hybrid (Keyword + Vector) search.
        
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
        filter: Filter=None
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
        collection = self._client.collections.get(collection_name)
        response = collection.query.hybrid(query=request,
                                           query_properties=query_properties,
                                           filters=filter,
                                           vector=query_vector,
                                           alpha=alpha,
                                           limit=limit,
                                           return_metadata=MetadataQuery(score=True, distance=True),
                                           return_properties=return_properties)
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
            self._client.collections.create(name=collection_name,
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
                         error_threshold: float=0.01,
                         vector_property: str='content_embedding', 
                         unique_id_field: str='doc_id',
                         properties: list[Property]=None,
                         collection_description: str=None,
                         **kwargs
                         ) -> dict:
        '''
        Batch function for fast indexing of data onto Weaviate cluster. 
        
        Args
        ----
        data: list[dict]
            List of dictionaries where each dictionary represents a document.
        collection_name: str
            Name of the collection to index data into.
        error_threshold: float=0.01
            Threshold for error rate during batch upload. This value is a percentage of the total data
            that the end user is willing to tolerate as errors. If the error rate exceeds this threshold,
            the batch job will be aborted.
        vector_property: str='content_embedding'
            Name of the property that contains the vector representation of the document.
        unique_id_field: str='doc_id'
            Name of the unique identifier field in the document.
        properties: list[Property]=None
            List of properties to create the collection with. Required if collection does not exist.
        collection_description: str=None
            Description of the collection. Optional parameter.
        
        Returns
        -------
        dict
            Dictionary containing error information if any with the following keys: 
            ['num_errors', 'error_messages', 'doc_ids']
        '''
        self._connect()
        if not self._client.collections.exists(collection_name):
            print(f'Collection "{collection_name}" not found on host, creating Collection first...')
            if properties is None:
                raise ValueError(f'Tried to create Collection <{collection_name}> but no properties were provided.')
            self.create_collection(collection_name=collection_name, 
                                   properties=properties,
                                   description=collection_description,
                                   **kwargs)
            self._client.close()

        self._connect()
        error_threshold_size = int(len(data) * error_threshold)
        collection = self._client.collections.get(collection_name)

        start = time.perf_counter()
        completed_job = True
        
        with collection.batch.dynamic() as batch:
            for doc in tqdm(data):
                batch.add_object(properties={k:v for k,v in doc.items() if k != vector_property},
                                 vector=doc[vector_property])
                if batch.number_errors > error_threshold_size:
                    print('Upload errors exceed error_threshold...')
                    completed_job = False
                    break 
        end = time.perf_counter() - start
        print(f'Processing finished in {round(end/60, 2)} minutes.')
        
        failed_objects = collection.batch.failed_objects
        if any(failed_objects):
            error_messages = [obj.message for obj in failed_objects]
            doc_ids = [obj.object_.properties.get(unique_id_field, 'Not Found') for obj in failed_objects] 
        else:
            error_messages, doc_ids = [], []
        error_object = {'num_errors':batch.number_errors, 
                        'error_messages': error_messages,
                        'doc_ids': doc_ids}
        if not completed_job:
            print(f'Batch job failed. Review errors using these keys: {list(error_object.keys())}')
            return error_object
        if batch.number_errors > 0:
                print(f'Batch job completed with {batch.number_errors} errors.  Review errors using these keys: {list(error_object.keys())}')
        else:
            print('Batch job completed with zero errors.')
        return error_object
            

@dataclass
class SearchFilter(Filter):

    '''
    Simplified interface for constructing a Filter object.

    Args
    ----
    property : str
        Property to filter on.
    query_value : str
        Query value to filter on.
    '''
    property: str
    query_value: str

    def exact_match(self):
        return self.by_property(self.property).equal(self.query_value)
    
    def fuzzy_match(self):
        return self.by_property(self.property).like(f'*{self.query_value}*')