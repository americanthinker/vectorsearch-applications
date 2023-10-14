from typing import Tuple, List, IO, Union, Dict
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk, parallel_bulk
from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import os

class OpenSearchClient(OpenSearch):
    '''
    Base Class for connecting with and building indexes on Opensearch host.  
    Included methods allow for keyword, semantic, and image indexing. All  
    python Opensearch API methods are available through this Class.

    Args
    -----
     hosts : str
        Full DNS address, prefixed by https:// and suffixed with host port.  

     http_auth : Tuple
         Tuple of strings: username and password combination.

     use_ssl : bool=True
        Flag indicating whether or not to use SSL for verification.  

     timeout : int=30
        Number of seconds to try connecting to Opensearch before exiting system.

    '''
    def __init__(self,
                 model_name_or_path: str=None,
                 hosts: List[dict]=[{"host": "localhost", "port": 9200}],
                 http_auth: Tuple[str, str]=('admin', 'admin'),
                 use_ssl: bool = True,
                 verify_certs = False,
                 ssl_assert_hostname = False,
                 ssl_show_warn = False,
                 timeout: int=30):
        super().__init__(hosts=hosts,
                         http_auth=http_auth,
                         use_ssl=use_ssl,
                         verify_certs = verify_certs,
                         ssl_assert_hostname = ssl_assert_hostname,
                         ssl_show_warn = ssl_show_warn,
                         timeout=timeout)
        self.timeout = timeout
        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(self.model_name_or_path) if self.model_name_or_path else None
        self.source_fields = ['content','group_id','doc_id','episode_url', 'episode_num', 'video_id','length','publish_date','thumbnail_url','title','views']
        

    def _create_index(self,
                     index_name: str,
                     body_template: dict, 
                     number_of_shards: int=None,
                     semantic_index: bool=False,
                     semantic_vector_dims: int=None,
                     ):
        '''
        Creates a new index on Elasticsearch host.

        Args
        -----
        index_name : str
            Name of document index to create. For reference see:
            https://www.elastic.co/blog/how-many-shards-should-i-have-in-my-elasticsearch-cluster

        number_of_shards : int=3
            Number of shards to create for index. 

        image_index : bool=False
            Flag indicating whether the index being created is for images or not.  
            If True, image_vector_dims is a required input.

        image_vector_dims : int=None
            Number of dimensions of image_vector.  Only required if image_index == True.

        semantic_index : bool=False
            Flag indicating whether the index being created is for semantic search or not.  
            If True, semantic_vector_dims is a required input.

        semantic_vector_dims : int=None
            Number of dimensions of content_embedding.  Only required if semantic_index == True.

        udf_mapping : dict=None
            User defined mapping for creating mapping fields for indexing.

        Returns
        --------
            None.  Creates index on Elasticsearch host.

        '''
        if semantic_index and not isinstance(semantic_vector_dims, int):
            raise ValueError(f'Are you trying to create a Semantic Index?  If so, semantic_vector_dims must be a postive integer.  \
                             "{semantic_vector_dims}" of type {type(semantic_vector_dims)} was passed.')

        if semantic_index:
            body_template['settings']['index']['knn'] = True
            body_template['mappings']['properties']['content_embedding'] = {"type": "knn_vector", "dimension": semantic_vector_dims}
        if number_of_shards:
            body_template['settings']['shards'] = number_of_shards

        self.indices.create(index=index_name, body=body_template)


    def _doc_generator(self,
                       index_data: List[dict],
                       index_name: str,
                       semantic_index: bool=False,
                       ) -> dict: 
        """
        For each dict in index_data converts meta data into a single doc accorindg to the index type. 
        This function is passed into the bulk() helper to push multiple docs onto an Opnesearch host
        index concurrently.
        """
        for dic in index_data:
            try:
                doc = {"_index": index_name}
                for k in dic:
                    if k != 'content_embedding':
                        doc[k] = dic[k]
                if semantic_index:
                    doc['content_embedding']=dic['content_embedding']
                yield doc
                
            except KeyError as e:
                logger.info(f'Key Error on this dict: {dic}')
                logger.info(e.args[0])
                continue
                
    def document_indexer(self,  
                        index_name: str, 
                        data: Union[IO[str], List[dict]],
                        body_template: dict=None,
                        semantic_index: bool=False,
                        chunk_size: int=2000,
                        number_of_shards: int=None,
                        update: bool=False,
                        ) -> None:
                     
        '''
        Primary user interface for building indexes from raw data.  
          - Data is read either from file or from memory
          - If index does not already exist, index will be created automatically.  
          - Function checks for image, semantic, udf, or regular keyword index type.
          - Docs are bulk (concurrent processing) indexed according to chunk_size.

        Args
        -----
        index_name : str
            Name of index to be created. 

        data : str | List[dict]
            Path to preprocessed .txt file, or in-memory list of dictionaries
            formatted through the Class DocumentProcessor. 
        
        image_index : bool=False
            Flag indicating if an image index is to be created.
        
        semantic_index : bool=False
            Flag indicating if a semantic index is to be created.
        
        udf_mapping : str=None
            User Defined Mapping = JSON/dictionary format.  Gives ends user the 
            ability to pass in a customized mapping for indexing. 

        chunksize : int=2000
            Number of documents in one chunk to push to Elasticsearch index.
            Indexing speeds vary by multiple factors so optimum chunk size
            must be empirically validated. If creating a semantic index, due 
            to their large size a value over 1,000 is not permitted. 

        number_of_shards : int=5
            If creating index from scratch (default), allows user to set 
            # of shards for index. Ignored if "update" equals True. 

        update : bool=False
            If True, function will update an already existing index.  
            Default will create a new index on execution.

         Returns
        ---------
         Does not return an object.  Index is built in place on target 
         Elasticsearch host.
        
        '''
        
        # if semantic_index:
        #     if chunk_size > 1000:
        #         chunk_size = 1000

        if isinstance(data, str):
            with open(data) as f:
                logger.info("Loading data from disk.")
                index_data = [json.loads(line) for line in f.readlines()]
        elif isinstance(data, list):
            index_data = data
        elif isinstance(data, dict):
            index_data = [data]

        semantic_vector_dims = len(index_data[0]['content_embedding']) if semantic_index else None

        if not update:
            try:
                self._create_index(index_name=index_name, 
                                   body_template=body_template,
                                   number_of_shards=number_of_shards,
                                   semantic_index=semantic_index, 
                                   semantic_vector_dims=semantic_vector_dims,
                                   )
                logger.info(f"The ** {index_name} ** index was created")

            #TODO: Find OpenSearch Exceptions 
            except Exception as e:
                logger.info(e)

        NUM_DOCS_INDEXED = len(index_data)
        logger.info(f"The # of documents to be indexed = {NUM_DOCS_INDEXED}")

        progress = tqdm(unit="Docs Indexed", total=NUM_DOCS_INDEXED)
        count = 0
        for success, _ in parallel_bulk(client=self, 
                                        actions=self._doc_generator(index_data=data,
                                                                    index_name=index_name,
                                                                    semantic_index=semantic_index),
                                        thread_count=os.cpu_count()*2, 
                                        chunk_size=chunk_size):
            try:
                progress.update(success)
                count += success
            except Exception as e:
                logger.info(e)
                
        logger.info(f'Number of docs indexed: {count}')
        
    def show_indexes(self, index_name: str=None):
        print(self.cat.indices(index=index_name, params={'v':'true'}))
    
    def keyword_search(self, query: str, index: str, size: int=10, return_raw: bool=False):
        '''
        Executes basic keyword search functionality.
        '''
        body = {
                # "_source": ['title', 'episode_num', 'episode_url', 'content'], 
                "_source": self.source_fields, 
                "size": size,
                "query": {
                    "bool": {
                        "must": {
                            "match": {"content": query,}
                                },
                            # "filter": {"bool": {"must_not": {"match_phrase": {"content": "Vishal"}}}},
                        },
                    },            
                }
        response = self.search(body=body, index=index)
        if return_raw: 
            return response 
        else: return response['hits']['hits']

    def vector_search(  self,
                        query: str, 
                        index: str, 
                        size: int=10,
                        k: int=10,
                        return_raw: bool=False
                        ) -> Dict[str,str]:
        '''
        Executes basic vector search functionality.
        '''
        if isinstance(self.model, SentenceTransformer):
            query_embedding = self.model.encode(query).tolist()

        body={  
                # "_source": ['title', 'episode_id', 'group_id', 'episode_num', 'episode_url', 'mp3_url', 'content'],
                "_source": self.source_fields, 
                "size": size,
                "query": 
                {"knn": {"content_embedding": {"vector": query_embedding, "k": k}}},
            }
        response = self.search(body=body, index=index)
        if return_raw: 
            return response 
        else: return response['hits']['hits']

    def hybrid_search(  self,
                        query: str, 
                        kw_index: str, 
                        vec_index: str,
                        kw_size: int=25,
                        vec_size: int=25,
                        dedup_results: bool=True,
                        return_raw: bool=False
                        ) -> Dict[str,str]:
        '''
        Executes a keyword search and a vector search.  Results are 
        naively combined into a single list and results are deduplicated.
        '''

        kw_result = self.keyword_search(query, kw_index, kw_size, return_raw)
        vec_result = self.vector_search(query, vec_index, vec_size, return_raw=return_raw)

        #interleave results of both searches into a single list
        hybrid_result = self._interleave_results(kw_result, vec_result)
        
        #remove duplicate results if dedup is True (default)
        if dedup_results:
            hybrid_result = self._deduplicate_results(hybrid_result)

        return hybrid_result
    
    def _interleave_results(self, 
                            kw_result: List[dict], 
                            vec_result: List[dict]
                            ) -> List[dict]:
        hybrid_result = []
        max_len = max(len(kw_result), len(vec_result))
        for i in range(max_len):
            if i < len(kw_result):
                hybrid_result.append(kw_result[i])
            if i < len(vec_result):
                hybrid_result.append(vec_result[i])
        return hybrid_result
    
    def _deduplicate_results(self, list_of_hits: List[dict]) -> List[dict]:
        '''
        Given a list of hits from a hybrid search call, returns a list of unique hits.
        '''
        unique_hits = {}
        for hit in list_of_hits:
            doc_id = hit['_source']['doc_id']
            if doc_id not in unique_hits:
                unique_hits[doc_id] = hit
        return list(unique_hits.values())
    
    def parse_content_from_response(self, list_of_hits: List[dict]) -> List[str]:
        '''
        Given a list of hits from a search call, returns a list of content strings.
        '''
        return [hit['_source']['content'] for hit in list_of_hits]