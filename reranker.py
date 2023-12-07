from sentence_transformers import CrossEncoder
from torch.nn import Sigmoid
from typing import List, Union
import numpy as np
from loguru import logger

class ReRanker(CrossEncoder):
    '''
    Cross-Encoder models achieve higher performance than Bi-Encoders, 
    however, they do not scale well to large datasets. The lack of scalability
    is due to the underlying cross-attention mechanism, which is computationally
    expensive.  Thus a Bi-Encoder is best used for 1st-stage document retrieval and 
    a Cross-Encoder is used to re-rank the retrieved documents. 

    https://www.sbert.net/examples/applications/cross-encoder/README.html
    '''

    def __init__(self, 
                 model_name: str='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 **kwargs
                 ):
        super().__init__(model_name=model_name, 
                         **kwargs) 
        self.model_name = model_name
        self.score_field = 'cross_score'
        self.activation_fct = Sigmoid()

    def _cross_encoder_score(self, 
                             results: List[dict], 
                             query: str, 
                             hit_field: str='content',
                             apply_sigmoid: bool=True,
                             return_scores: bool=False
                             ) -> Union[np.array, None]:
        '''
        Given a list of hits from a Retriever:
            1. Scores hits by passing query and results through CrossEncoder model. 
            2. Adds cross-score key to results dictionary. 
            3. If desired returns np.array of Cross Encoder scores.
        '''
        activation_fct = self.activation_fct if apply_sigmoid else None
        #build query/content list
        cross_inp = [[query, hit[hit_field]] for hit in results]
        #get scores
        cross_scores = self.predict(cross_inp, activation_fct=activation_fct)
        for i, result in enumerate(results):
            result[self.score_field]=cross_scores[i]

        if return_scores:return cross_scores

    def rerank(self, 
               results: List[dict], 
               query: str, 
               top_k: int=10, 
               apply_sigmoid: bool=True,
               threshold: float=None
               ) -> List[dict]:
        '''
        Given a list of hits from a Retriever:
            1. Scores hits by passing query and results through CrossEncoder model. 
            2. Adds cross_score key to results dictionary. 
            3. Returns reranked results limited by either a threshold value or top_k.
        
        Args:
        -----
        results : List[dict]
            List of results from the Weaviate client
        query : str
            User query
        top_k : int=10
            Number of results to return
        apply_sigmoid : bool=True
            Whether to apply sigmoid activation to cross-encoder scores.  If False, 
            returns raw cross-encoder scores (logits).
        threshold : float=None
            Minimum cross-encoder score to return. If no hits are above threshold, 
            returns top_k hits.
        '''
        # Sort results by the cross-encoder scores
        self._cross_encoder_score(results=results, query=query, apply_sigmoid=apply_sigmoid)

        sorted_hits = sorted(results, key=lambda x: x[self.score_field], reverse=True)
        if threshold or threshold == 0:
            filtered_hits = [hit for hit in sorted_hits if hit[self.score_field] >= threshold]
            if not any(filtered_hits):
                logger.warning(f'No hits above threshold {threshold}. Returning top {top_k} hits.')
                return sorted_hits[:top_k]
            return filtered_hits
        return sorted_hits[:top_k]