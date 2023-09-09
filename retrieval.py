from qdrant_client import QdrantClient
import openai

class Retriever:

    def __init__(self, model_type: str, location: str='http://localhost:6333'):
        # Initialize encoder model
        self.model_type = model_type
        # self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(location)

    def search(self, query: str, collection: str, limit: int=5, return_all: bool=False):
        '''
        First-stage retrieval of documents.  Query is vectorized and compared against 
        other vectors in database.  Distance metric (cosine) is used to return closest
        results in vector space.
        '''
        # Convert text query into vector
        # vector = self.model.encode(query).tolist()
        results = openai.Embedding.create(input=[query], engine=self.model_type)
        vector = results['data'][0]['embedding']
        
        search_result = self.qdrant_client.search(
                                            collection_name=collection,
                                            query_vector=vector,
                                            query_filter=None,  
                                            limit=limit)
        if return_all: 
            return search_result
        else: 
            payloads = [hit.payload for hit in search_result]
            return payloads
