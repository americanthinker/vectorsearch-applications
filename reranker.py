from sentence_transformers import CrossEncoder
from typing import List, Union
import numpy as np
from loguru import logger

class ReRanker:
    '''
    Cross-Encoder models achieve higher performance than Bi-Encoders, 
    however, they do not scale well to large datasets. The lack of scalability
    is due to the underlying cross-attention mechanism, which is computationally
    expensive.  Thus a Bi-Encoder is best used for 1st-stage document retrieval and 
    a Cross-Encoder is used to re-rank the retrieved documents. 

    https://www.sbert.net/examples/applications/cross-encoder/README.html
    '''

    def __init__(self, model_name: str='cross-encoder/ms-marco-MiniLM-L-6-v2', local_files: bool=False):
        self.model_name = model_name
        self.model = CrossEncoder(self.model_name, automodel_args={'local_files_only':local_files})
        self.score_field = 'cross_score'

    def _cross_encoder_score(self, 
                             results: List[dict], 
                             query: str, 
                             return_scores: bool=False
                             ) -> Union[np.array, None]:
        '''
        Given a list of hits from a Retriever:
            1. Scores hits by passing query and results through CrossEncoder model. 
            2. Adds cross-score key to hits dictionary. 
            3. If desired returns np.array of Cross Encoder scores.
        '''
        
        #build query/content list
        cross_inp = [[query, hit['_source']['content']] for hit in results]
        #get scores
        cross_scores = self.model.predict(cross_inp)
        for i, result in enumerate(results):
            result[self.score_field]=cross_scores[i]

        if return_scores:return cross_scores

    def rerank(self, 
               results: List[dict], 
               query: str, 
               top_k: int=10, 
               threshold: float=None
               ) -> List[dict]:
        # Sort results by the cross-encoder scores
        self._cross_encoder_score(results=results, query=query)

        sorted_hits = sorted(results, key=lambda x: x[self.score_field], reverse=True)
        if threshold or threshold == 0:
            filtered_hits = [hit for hit in sorted_hits if hit[self.score_field] >= threshold]
            if not any(filtered_hits):
                logger.warning(f'No hits above threshold {threshold}. Returning top {top_k} hits.')
                return sorted_hits[:top_k]
            return filtered_hits
        return sorted_hits[:top_k]