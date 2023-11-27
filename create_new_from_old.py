subset_doc_ids = list(golden_dataset.corpus.keys())[:100]
corpus = {}
for doc_id in subset_doc_ids:
    corpus[doc_id] = golden_dataset.corpus[doc_id]
len(corpus)

reverse_rel = {v[0]:k for k,v in golden_dataset.relevant_docs.items()}
reverse_rel = {k:v for k,v in reverse_rel.items() if k in subset}
len(reverse_rel)

relevant_docs = {}
for doc_id, query_num in reverse_rel.items():
    relevant_docs[query_num] = [doc_id]
len(relevant_docs)

queries = {query_id : golden_dataset.queries[query_id] for query_id in relevant_docs}
len(queries)

golden_100 = EmbeddingQAFinetuneDataset(queries=queries, corpus=corpus, relevant_docs=relevant_docs)

golden_100.save_json('./data/golden_100.json')