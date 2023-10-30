from preprocessing import FileIO, Vectorizor
from opensearch_interface import OpenSearchClient
from index_templates import youtube_body

import os
import sys
import time
from rich import print
from dotenv import load_dotenv
load_env=load_dotenv('./.env', override=True)

model_path = 'sentence-transformers/all-MiniLM-L6-v2'
model_path_on_disk = os.environ['ST_MODEL_PATH']
osclient = OpenSearchClient(model_name_or_path=model_path)
print(osclient.show_indexes())
query = sys.argv[1]
index_name = 'impact-theory-minilm-196'
kw_response = osclient.vector_search(query=query, index=index_name, size=5)
print(kw_response)