import streamlit as st
from opensearch_interface import OpenSearchClient
from sentence_transformers import SentenceTransformer
from reranker import ReRanker
from loguru import logger 
import templates

osclient = OpenSearchClient()
logger.info(osclient.show_indexes())

kw_index = 'kw-538-testrun'
vec_index = 'semantic-538-testrun'
model_name = 'all-minilm-l6-v2'
model = SentenceTransformer(model_name)
reranker = ReRanker()

def main():
    st.write(templates.load_css(), unsafe_allow_html=True)
    
    # st.title('Chat with the 538 YouTube Podcast:')
    st.image('./assets/538_logo.png', width=400)
    st.subheader("Chat with the 538 podcast: ")
    st.write('\n')
    col1, col2 = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')
        st.subheader("Search Results")

    if query:
        #execute kw and vector search call to OpenSearch 
        hybrid_response = osclient.hybrid_search(query, kw_index, vec_index, model, kw_size=25, vec_size=25)

        #rerank response from search call using CrossEncoder
        final_response = reranker.rerank(hybrid_response, query)

        #display results using imported css templates
        for i, res in enumerate(final_response):
            col1, col2 = st.columns([7, 3], gap='large')
            hit = res['_source']
            image = hit['thumbnail_url']
            title = (hit['title'])
            with col1:
                st.write(templates.search_result(i=i, 
                                                url=hit['show_link'],
                                                title=title,
                                                content=hit['content'], 
                                                length=hit['length']),
                        unsafe_allow_html=True)
                st.write('\n\n')
            with col2:
                st.image(image, caption=title, width=200, use_column_width=False)
        #for debugging purposes
        logger.info(final_response[0])

if __name__ == '__main__':
    main()