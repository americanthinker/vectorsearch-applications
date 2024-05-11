import sys
sys.path.append("../")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True)

import os
import uuid
from time import sleep
from loguru import logger
from typing import Generator, Any
import streamlit as st
from dotenv import load_dotenv

from tiktoken import Encoding, get_encoding
from src.llm.prompt_templates import (
    huberman_system_message,
    question_answering_prompt_series,
    generate_prompt_series,
    ui_introduction_message
)
from src.llm.llm_utils import load_azure_openai
from app_functions import validate_token_threshold
from src.reranker import ReRanker
from src.database.weaviate_interface_v4 import WeaviateWCS
from src.database.database_utils import get_weaviate_client
from src.conversation import Conversation, Message

abs_path = os.path.abspath("../../")
sys.path.append(abs_path)

## Page Configuration
st.set_page_config(
    page_title="Huberman Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None,
)

## Constants
collection_name = 'Huberman_minilm_256'
turbo = 'gpt-3.5-turbo-0125'
claude = 'claude-3-haiku-20240307'
anthro_api_key = os.getenv('ANTHROPIC_API_KEY')
data_path = '../data/huberman_labs.json'
# content_data_path = '../impact-theory-newft-256.parquet'
UI_CONVERSATION_KEY = "ui_conversation"
CONTEXT_CONVERSATION_KEY = "context_conversation"
MESSAGE_BUILDER_KEY = "message_builder"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-base-en"
INDEX_NAME = "odin-bge-768-feb5"
TITLES = "titles"
PAGES = "pages"
huberman_icon = "/workspaces/vectorsearch-applications/answer_key/assets/huberman_logo.png"
uplimit_icon = '/workspaces/vectorsearch-applications/answer_key/assets/uplimit_logo.jpg'


## RETRIEVER
@st.cache_resource
def get_retriever() -> WeaviateWCS:
    return get_weaviate_client(model_name_or_path=EMBEDDING_MODEL_PATH)

# Cache reranker
@st.cache_resource
def get_reranker() -> ReRanker:
    return ReRanker()

# Cache LLM
@st.cache_resource
def get_llm(model_name: str='gpt-35-turbo'):
    return load_azure_openai(model_name=model_name)


## Cache encoding model
@st.cache_resource
def get_encoding_model(model_name) -> Encoding:
    return get_encoding(model_name)

## Get cached resouces
reranker = get_reranker()
llm = get_llm()
encoding = get_encoding_model("cl100k_base")
retriever = get_retriever()

## Display fields
LLM_CONTENT_FIELD = "content"
DSIPLAY_CONTENT_FIELD = "content"
app_display_fields = retriever.return_properties 
logger.info(app_display_fields)


# Run the chat interface within Streamlit
def run_chat_interface():
    """
    Populates the chat area in the UI, sets the chat controls, and handles user input.
    
    Args:
        None

    Returns:
        None

    """
    populate_chat_area(st.session_state[UI_CONVERSATION_KEY])

    # Chat controls
    clear_button = st.button("Clear Chat History")
    user_input = st.chat_input("Ask something:")

    # Clear chat history
    if clear_button:
        set_new_conversations()
        st.rerun()

    # Handle user input and generate assistant response
    if user_input or st.session_state.streaming:
        with st.chat_message("user", avatar=uplimit_icon):
            st.write(user_input)
        process_user_input(user_input)
        # After processing user input, rerun the app. 
        st.rerun()


def populate_chat_area(conversation: Conversation):
    """
    Iterate over a Conversation and display the chat messages in the chat area.

    Args:
        conversation (Conversation): The conversation object containing the chat history.

    Returns:
        None

    """
    for msg in conversation.queue_to_list():
        # Use this call to pass in the Odin icon
        if msg["role"] == "assistant":
            with st.chat_message(name="assistant", avatar=huberman_icon):
                st.write(msg["content"])
        else:
            with st.chat_message(msg["role"], avatar=uplimit_icon):
                st.write(msg["content"])


def set_new_conversations():
    """
    Create a new conversation.

    Args:
        user (str, optional): The user for whom the conversation is created. Defaults to "user".

    Returns:
        Conversation: The newly created conversation object.
    """
    st.session_state[UI_CONVERSATION_KEY] = Conversation(
        conversation_id=str(uuid.uuid4()),
        system_message=Message(role="assistant", content=ui_introduction_message),
    )
    st.session_state[CONTEXT_CONVERSATION_KEY] = Conversation(
        conversation_id=str(uuid.uuid4()),
        system_message=Message(role="system", content=huberman_system_message),
    )


def process_user_input(user_input):
    """Process the user input and generate the assistant response."""
    if user_input:
        # 1. Run rag search
        results = retriever.hybrid_search(
            user_input, collection_name=collection_name, return_properties=app_display_fields, limit=200
        )

        # 2. Rerank search results using semantic reranker
        reranked_results = reranker.rerank(
            results, user_input, apply_sigmoid=False, top_k=5
        )

        # 3. Validate token threshold
        valid_results = validate_token_threshold(
            reranked_results,
            question_answering_prompt_series,
            query=user_input,
            tokenizer=encoding,
            token_threshold=6000,
            content_field=LLM_CONTENT_FIELD
        )

        # 4. Generate context series
        context_series = generate_prompt_series(user_input, valid_results, 1)
        
        # 5. Add messages to conversations
        st.session_state[UI_CONVERSATION_KEY].add_message(
            Message(role="user", content=user_input)
        )
        st.session_state[CONTEXT_CONVERSATION_KEY].add_message(
            Message(role="system", content=context_series)
        )

        # debugging purposes
        logger.info(st.session_state[UI_CONVERSATION_KEY].queue_to_list())

        # 6. Generate assistant response
        with st.chat_message(name="assistant", avatar=huberman_icon):
            gpt_answer = st.write_stream(
                chat(
                    user_message=context_series,
                    max_tokens=1000
                )
            )

        ref_docs = list(
            set(list(zip(st.session_state[TITLES], st.session_state[PAGES])))
        )
        if any(ref_docs):
            with st.expander("Reference Documents", expanded=False):
                for i, doc in enumerate(ref_docs, start=1):
                    st.markdown(f"{i}. **{doc[0]}**: &nbsp; &nbsp; page {doc[1]}")
            st.session_state[TITLES], st.session_state[PAGES] = [], []

        # 7. Add assistant response to the conversation
        st.session_state[UI_CONVERSATION_KEY].add_message(
            Message(role="assistant", content=gpt_answer)
        )
        #     st.session_state.generator = gpt_answer
        #     st.session_state.streaming = True
        #     st.rerun()
        # else:
        #     update_assistant_response()


# Generate chat responses using the OpenAI API
def chat(
    user_message: str,
    max_tokens: int=250,
    temperature: float=0.5,
 ) -> Generator[Any, Any, None]:
    """Generate chat responses using an LLM API.
    Stream response out to UI.
    """
    completion = llm.chat_completion(huberman_system_message,
                                     user_message=user_message,
                                     temperature=temperature,
                                     max_tokens=max_tokens,
                                     stream=True)

    full_json = []
    for chunk in completion:
        sleep(0.05)
        if any(chunk.choices):
            content = chunk.choices[0].delta.content
            if content:
                full_json.append(content)
                yield content

    answer = "".join(full_json)
    # logger.info(answer)
    st.session_state[MESSAGE_BUILDER_KEY] = {"role": "assistant", "content": answer}

# Main function to run the Streamlit app
def main():
    """Main function to run the Streamlit app."""
    st.markdown(
        """<style>.block-container{max-width: 66rem !important;}</style>""",
        unsafe_allow_html=True,
    )
    st.title("Chat with the Huberman Lab podcast")
    st.markdown("---")

    # Session state initialization
    if UI_CONVERSATION_KEY not in st.session_state:
        set_new_conversations()
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if TITLES not in st.session_state:
        st.session_state[TITLES] = []
    if PAGES not in st.session_state:
        st.session_state[PAGES] = []

    run_chat_interface()

if __name__ == "__main__":
    main()
