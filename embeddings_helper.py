import openai
import pandas as pd

def get_embedding(text_to_embed: str, model: str="text-embedding-ada-002"):
	'''
    Given a string of text, calls Open AI API and returns an embedding. 
    '''
	response = openai.Embedding.create(model= model, input=[text_to_embed])
	
	return response["data"][0]["embedding"]


def get_text_stats(df: pd.DataFrame, text_column: str):
    return df[text_column].apply(lambda x: len(str(x).split()))
    