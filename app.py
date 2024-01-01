from tiktoken import get_encoding
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

###################################
#### CLASS NAME AND DATA PATHS ####
#### WILL CHANGE USER TO USER  ####
###################################
class_name = 'Fine_tuned_minilm_256'
data_path = './data/impact_theory_data.json'
content_data_path = '../impact-theory-newft-256.parquet'
embedding_model_path = './models/finetuned-all-MiniLM-L6-v2-300/'
###################################

## RETRIEVER
client = WeaviateClient(os.environ['WEAVIATE_API_KEY'], 
                        os.environ['WEAVIATE_ENDPOINT'], 
                        model_name_or_path=embedding_model_path)
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
#loads cache of content data
content_cache = load_content_cache(content_data_path)
#loads data for property extraction
data = load_data(data_path)
#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
    #set sidebar action    
    with st.sidebar:
        guest = st.selectbox('Select Guest', options=guest_list, index=None, placeholder='Select Guest')

    subheader_entry = 'Impact Theory'
    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the {subheader_entry}: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

    if query:
        logger.info(client)
        with st.spinner('Searching...'):
            filter = WhereFilter(path=['guest'], operator='Equal', valueText=guest).todict() if guest else None
            hybrid_response = client.hybrid_search(query, properties=['content', 'summary', 'title'], class_name=class_name,\
                                                   alpha=0.3,display_properties=display_properties, where_filter=filter, \
                                                   limit=200)
            logger.info(f'RESPONSE: {hybrid_response}')
            ranked_response = reranker.rerank(hybrid_response, query, top_k=3)
            
            for resp in ranked_response:
                pre = sum([len(d['content']) for d in ranked_response])
                logger.info(f'PRE: {pre}')
                split_id = resp['doc_id'].split('_')
                doc_int_id = split_id[-1]
                doc_idd = '_'.join(split_id[:-1])
                doc_idd += '_'
                pre = int(doc_int_id) - 1
                post = int(doc_int_id) + 1
                context1, context2, context3 = content_cache.get(doc_idd + str(pre), ''), \
                                               resp['content'], \
                                               content_cache.get(doc_idd + str(post),'')
                full_context = ' '.join([context1, context2, context3])
                resp['content'] = full_context
                logger.info(f'POST: {len(resp["content"])}')
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

            #creates container for LLM response
            chat_container, response_box = [], st.empty()
            #streaming LLM call
            for resp in gpt.get_chat_completion(prompt=prompt,
                                                system_message=question_answering_system,
                                                max_tokens=250, 
                                                stream = True,
                                                show_response=True
                                                ):
                try:
                    #inserts chat stream from LLM 
                    with response_box:
                        content = resp.choices[0].delta.content
                        if content:
                            chat_container.append(content)
                            result = "".join(chat_container).strip()
                            st.write(f'{result}')
                
                except Exception as e:
                    print(e)
                    continue
        logger.info(result)
        st.subheader("Search Results")
        for i, hit in enumerate(valid_response):
            col1, col2 = st.columns([7, 3], gap='large')
            episode_url = hit['episode_url']
            title = hit['title']
            show_length = hit['length']
            time_string = convert_seconds(show_length)

            #break out search reults into two columns
            #column 1 = search result
            with col1:
                st.write(search_result( i=i, 
                                        url=episode_url,
                                        guest=hit['guest'],
                                        title=title,
                                        content=content_cache[hit['doc_id']], 
                                        length=time_string),
                        unsafe_allow_html=True)
                st.write('\n\n')

            #column 2 = thumbnail image
            with col2:
                image = hit['thumbnail_url']
                st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                st.markdown(f'<p style="text-align": right;"><b>Guest: {hit["guest"]}</b>', unsafe_allow_html=True)
           
if __name__ == '__main__':
    main()