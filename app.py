from tiktoken import get_encoding
from weaviate_interface import WeaviateClient
from prompt_templates import question_answering_prompt, question_answering_system
from openai_interface import GPT_Turbo
from app_features import convert_seconds, generate_prompt
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import css_templates
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
gpt = GPT_Turbo('gpt-3.5-turbo-16k')
## TOKENIZER
encoding = get_encoding("cl100k_base")

indexes = ['Impact_theory_minilm_256', 'wands-products']
index_name_mapper = {indexes[0]:'Impact Theory podcast', indexes[1]:'Wayfair Dataset'}

def main():
    st.write(css_templates.load_css(), unsafe_allow_html=True)
    
    with st.sidebar:
        index_options =  st.selectbox('Select Search Index', indexes)
        st.write('You selected:', index_options)

    subheader_entry = index_name_mapper[index_options]
    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the {subheader_entry}: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

    if query:
        # with st.spinner('Searching...'):
        hybrid_response = client._hybrid_search(query, class_name=index_options, alpha=0.3, limit=160)
        ranked_response = reranker.rerank(hybrid_response, query, top_k=6)
        prompt = generate_prompt(base_prompt=question_answering_prompt, query=query, results=ranked_response)
        print(prompt)
        import sys
        sys.exit()
        #execute chat call to OpenAI
        st.subheader("Response from Impact Theory (context)")
        with st.spinner('Generating Response...'):
            st.markdown("----")
            res_box = st.empty()
            report = []
            col_1, col_2 = st.columns([4, 3], gap='large')
            for resp in gpt.get_completion_from_messages(prompt=prompt,
                                                         system_message=question_answering_system,
                                                         max_tokens=250, 
                                                         stream = True,
                                                         show_response=True
                                                         ):
                try:
                    with col_1:
                        with res_box:
                            report.append(resp.choices[0].delta.get('content', '\n'))
                            result = "".join(report).strip()
                            # result = result.replace("\n", "")    
                            res_box.markdown(f'*{result}*')
                except Exception as e:
                    print(e)
                    continue

            # for i, resp in enumerate(response, 1):
            #     title = resp['_source']['title']
            #     st.text_area(label=title, value=resp['_source']['content'], height=250)
            #     st.write('---')

            # parsed_response = osclient.parse_content_from_response(final_response)
            # context = ' '.join(parsed_response)
            # prompt = f'''
            #          Using the following context enclosed in triple backticks, generate a response to the following question:
            #          Question: {query}
            #          Context: ```{context}```
            #          '''
            # response = gpt.get_completion_from_messages(prompt=prompt, temperature=0, max_tokens=500)
            # col1, _ = st.columns([7,3])
            # with col1:
            #     st.text_area('ChatGPT Response', response, height=150)
            #     st.write('\n\n\n')
        #display results using imported css templates
        st.subheader("Search Results")
        for i, res in enumerate(hybrid_response):
            col1, col2 = st.columns([7, 3], gap='large')
            hit = res['_source']
            image = hit['thumbnail_url']
            episode_url = hit['episode_url']
            title = (hit['title'])
            show_length = hit['length']
            time_string = convert_seconds(show_length)

            with col1:
                st.write(css_templates.search_result(i=i, 
                                                url=episode_url,
                                                episode_num=hit['episode_num'],
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