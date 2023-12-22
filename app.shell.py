from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

data_path = './data/impact_theory_data.json'

weaviate_api_key = os.environ['WEAVIATE_API_KEY']
weaviate_endpoint = os.environ['WEAVIATE_ENDPOINT']
openai_api_key = os.environ['OPENAI_API_KEY']
## RETRIEVER

retriever = WeaviateClient(weaviate_api_key, weaviate_endpoint)
retriever.display_properties.append('summary')

## RERANKER

reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

## LLM 

llm = GPT_Turbo()

## ENCODING

encoding = encoding_for_model('gpt-3.5-turbo-0613')

## INDEX NAME

index_name = "ImpactTheoryMinilm256"

## Alpha

alpha = 0.5

## SEARCH PROPERTIES

search_properties = ['title', 'content']

data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')
        temperature_input = st.slider('Temperature of LLM',0.0,2.0,1.0)

    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        if query:

            st.write('Hmmm...this app does not seem to be working yet.  Please check back later.')
            if guest:
                st.write(f'It looks like you selected {guest} as a filter.')
                guest_filter = WhereFilter(['guest'], operator='Equal', valueText=guest).todict()

            # make hybrid call to weaviate
            hybrid_response = retriever.hybrid_search(query, index_name, alpha=alpha, properties=search_properties,
                                                      where_filter=guest_filter)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response, query, apply_sigmoid=True, top_k=3)
            # filtered_ranked_response = [resp for resp in ranked_response if resp["cross_score"] >= 0.8]

            # validate token count is below threshold
            valid_response = validate_token_threshold(
                ranked_response, 
                question_answering_prompt_series, 
                query=query,
                tokenizer=encoding,
                token_threshold=4000, 
                verbose=True)

            # # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)
                        
            # # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                #creates container for LLM response
                chat_container, response_box = [], st.empty()
                
                # execute chat call to LLM
                for resp in llm.get_chat_completion(prompt=prompt,
                                                    temperature=temperature_input,
                                                    max_tokens=500,
                                                    show_response=True,
                                                    stream=True):
                    try:
                        #   inserts chat stream from LLM
                        with response_box:
                            content = resp.choices[0].delta.content
                            if content:
                                chat_container.append(content)
                                result = "".join(chat_container).strip()
                                st.write(f'{result}')
                    except Exception as e:
                        print(e)
                        continue
            if result == "I cannot answer the question given the context.":
                st.write("Please try writing another question or select a question from the list.")
            else:
                st.subheader("Search Results")
                for i, hit in enumerate(valid_response):
                    col1, col2 = st.columns([7, 3], gap='large')
                    image = hit["thumbnail_url"]
                    episode_url = hit["episode_url"]
                    title = hit["title"]
                    show_length = hit["length"]
                    time_string = convert_seconds(show_length)
                    with col1:
                        st.write( search_result(  i=i, 
                                                    url=episode_url,
                                                    guest=hit['guest'],
                                                    title=title,
                                                    content=hit['content'], 
                                                    length=time_string),
                                unsafe_allow_html=True)
                        st.write('\n\n')
                    with col2:
                        # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
                        #             unsafe_allow_html=True)
                        st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()