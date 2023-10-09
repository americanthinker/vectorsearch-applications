import streamlit as st
from opensearch_interface import OpenSearchClient
from openai_interface import GPT_Turbo
from sentence_transformers import SentenceTransformer
from reranker import ReRanker
from loguru import logger 
import templates
import time
 
def convert_seconds(seconds: int):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))
     
osclient = OpenSearchClient()
gpt = GPT_Turbo()
logger.info(osclient.show_indexes())

kw_index = 'kw-impact-theory'
vec_index = 'semantic-impact-theory'
model_name = 'all-minilm-l6-v2'
model = SentenceTransformer(model_name)
reranker = ReRanker()

def main():
    st.write(templates.load_css(), unsafe_allow_html=True)
    
    # st.title('Chat with the 538 YouTube Podcast:')
    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader("Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

    if query:
        #execute kw and vector search call to OpenSearch 
        # with st.spinner('Searching...'):
        hybrid_response = osclient.hybrid_search(query, kw_index, vec_index, model, kw_size=50, vec_size=50)

        #rerank response from search call using CrossEncoder
        final_response = reranker.rerank(hybrid_response, query, top_k=10)

        #for debugging purposes
        logger.info(final_response[0])

        #execute chat call to OpenAI
        st.subheader("Response from ChatGPT-3.5-Turbo")
        with st.spinner('Generating Response...'):
            parsed_response = osclient.parse_content_from_response(final_response)
            context = ' '.join(parsed_response)
            prompt = f'''
                     Using the following context enclosed in triple backticks, generate a response to the following question:
                     Question: {query}
                     Context: ```{context}```
                     '''
            response = gpt.get_completion_from_messages(prompt=prompt, temperature=0, max_tokens=500)
            col1, _ = st.columns([7,3])
            with col1:
                st.text_area('ChatGPT Response', response, height=150)
                st.write('\n\n\n')
        #display results using imported css templates
        st.subheader("Search Results")
        for i, res in enumerate(final_response):
            col1, col2 = st.columns([7, 3], gap='large')
            hit = res['_source']
            image = hit['thumbnail_url']
            episode_url = hit['episode_url']
            title = (hit['title'])
            length = hit['length']
            time_string = convert_seconds(length)
            logger.info(hit)
            with col1:
                st.write(templates.search_result(i=i, 
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