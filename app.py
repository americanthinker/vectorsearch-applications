from tiktoken import get_encoding
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import convert_seconds, generate_prompt_series, validate_token_threshold
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import css_templates
import json
import os

from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
## RETRIEVER
client = WeaviateClient(os.environ['WEAVIATE_API_KEY'], os.environ['WEAVIATE_ENDPOINT'])
if client.is_ready():
    logger.info('Weaviate is ready!')

## RERANKER
reranker = ReRanker()
## QA MODEL
gpt = GPT_Turbo('gpt-3.5-turbo-0613')
## TOKENIZER
encoding = get_encoding("cl100k_base")
## Display properties
display_properties = client.display_properties + ['summary']

indexes = ['Impact_theory_minilm_256', 'wands-products']
index_name_mapper = {indexes[0]:'Impact Theory podcast', indexes[1]:'Wayfair Dataset'}
data_path = './data/impact_theory_data.json'
with open(data_path, 'r') as f:
    data = json.load(f)
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    st.write(css_templates.load_css(), unsafe_allow_html=True)
    
    with st.sidebar:
        index_options =  indexes[0]
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    subheader_entry = index_name_mapper[index_options]
    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the {subheader_entry}: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

    if query:
        with st.spinner('Searching...'):
            filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest).todict() if guest else None
            hybrid_response = client.hybrid_search(query, class_name=index_options, alpha=0.3,\
                                                    display_properties=display_properties, where_filter=filter, limit=160)
            ranked_response = reranker.rerank(hybrid_response, query, top_k=5)
            valid_response = validate_token_threshold(ranked_response, 
                                                    question_answering_prompt_series, 
                                                    query=query,
                                                    tokenizer=encoding,
                                                    token_threshold=6000, 
                                                    verbose=True)
        prompt = generate_prompt_series(query, valid_response)
        logger.info('Prompt generated!')
        #execute chat call to OpenAI
        st.subheader("Response from Impact Theory (context)")
        with st.spinner('Generating Response...'):
            st.markdown("----")
            res_box = st.empty()
            report = []
            col_1, _ = st.columns([4, 3], gap='large')
            for resp in gpt.get_chat_completion(prompt=prompt,
                                                system_message=question_answering_system,
                                                max_tokens=250, 
                                                stream = True,
                                                show_response=True
                                                ):
                try:
                    with col_1:
                        with res_box:
                            print(resp)
                            content = resp.choices[0].delta.content
                            if content:
                                report.append(content)
                                result = "".join(report).strip()
                                # result = result.replace("\n", "")    
                                res_box.markdown(f'{result}')
                except Exception as e:
                    print(e)
                    continue

        st.subheader("Search Results")
        for i, hit in enumerate(valid_response):
            col1, col2 = st.columns([7, 3], gap='large')
            # hit = res['_source']
            image = hit['thumbnail_url']
            episode_url = hit['episode_url']
            title = (hit['title'])
            show_length = hit['length']
            time_string = convert_seconds(show_length)

            with col1:
                st.write(css_templates.search_result(i=i, 
                                                url=episode_url,
                                                episode_num=i,
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