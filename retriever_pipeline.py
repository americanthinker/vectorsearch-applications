from opensearch_interface import OpenSearchClient
from reranker import ReRanker
from rich import print
from typing import Literal, List
import tiktoken

def retrieve_pipeline(query: str, 
                      index_name: str,
                      search_type: Literal['kw', 'vector', 'hybrid'], 
                      retriever: OpenSearchClient, 
                      reranker: ReRanker,
                      tokenizer: tiktoken.core.Encoding,
                      kw_size: int=50,
                      vec_size: int=50,
                      top_k: int=4,
                      rerank_threshold: float=None,
                      token_threshold: int=4000,
                      return_text: bool=True,
                      verbose: bool=True
                      ) -> List[dict]:
     
    if search_type == 'kw':
        results = retriever.keyword_search(query=query, index=index_name, size=kw_size)
    elif search_type == 'vector':
        results = retriever.vector_search(query=query, index=index_name, size=vec_size)
    elif search_type == 'hybrid':
        results = retriever.hybrid_search(query=query, 
                                          kw_index=index_name, 
                                          vec_index=index_name, 
                                          kw_size=kw_size,
                                          vec_size=vec_size)
        
    reranked = reranker.rerank(results, query, top_k=top_k, threshold=rerank_threshold)
    text = ' '.join([r['_source']['content'] for r in reranked])
    token_count = len(tokenizer.encode_batch(text))
    if verbose:
        print(f'Total Initial Token Count: {token_count}')
    if token_count > token_threshold:
        print('Token count exceeds token count threshold, reducing size of returned results below token threshold')
        while token_count > token_threshold:
            num_results = len(reranked)
            reranked = reranked[:num_results-1]
            text = ' '.join([r['_source']['content'] for r in reranked])
            token_count = len(tokenizer.encode_batch(text))
        if verbose:
            print(f'Total Final Token Count: {token_count}')
    if return_text:
        return text
    return reranked
