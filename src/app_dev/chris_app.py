import sys
import os
abs_path = os.path.abspath("../../")
sys.path.append(abs_path)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True)

import uuid
from time import sleep
from loguru import logger
from typing import Generator, Any
import streamlit as st
from dotenv import load_dotenv
from src.llm.prompt_templates import (
    huberman_system_message,
    generate_prompt_series,
)
from app_functions import (
    validate_token_threshold,
    get_retriever,
    get_encoding_model,
    get_reranker,
    get_llm
)
from src.conversation import Conversation, Message
from src.app_dev.query import (
    CompletedQueryQueue,
    CompletedQuery,
    parse_context_results,
    parse_llm_response
)

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
UI_CONVERSATION_KEY = "ui_conversation"
LLM_CONTEXT_QUEUE = "llm_context_queue"
MESSAGE_BUILDER_KEY = "message_builder"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-base-en"
TITLES = "titles"
PAGES = "pages"
huberman_icon = "/workspaces/vectorsearch-applications/src/app_assets/huberman_logo.png"
uplimit_icon = '/workspaces/vectorsearch-applications/src/app_assets/uplimit_logo.jpg'


## Get cached resouces
reranker = get_reranker()
llm = get_llm(turbo)
llm_verbsoity = 1
encoding = get_encoding_model("cl100k_base")
retriever = get_retriever(EMBEDDING_MODEL_PATH)

## Display fields
LLM_CONTENT_FIELD = "content"
DSIPLAY_CONTENT_FIELD = "content"
app_display_fields = retriever.return_properties 

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
            system_message=Message(role="assistant", content="Welcome to the Huberman Lab podcast!"),
        )
    st.session_state[LLM_CONTEXT_QUEUE] = CompletedQueryQueue()


def process_user_input(user_input):
    """Process the user input and generate the assistant response."""
    if user_input:
        # 1. Submit user input with previous context to LLM for possible rewrite
        context_queue = st.session_state[LLM_CONTEXT_QUEUE]
        query_rewrite_prompt = context_queue.generate_prompt(user_input)
        llm_rewrite_response = llm.chat_completion(huberman_system_message,
            user_message=query_rewrite_prompt,
            temperature=0.5,
            max_tokens=1000)
        
        llm_rewrite_query_type, llm_rewrite_query_text = parse_llm_response(llm_rewrite_response)

        # 2. Run rag search
        results = retriever.hybrid_search(
            llm_rewrite_query_text, collection_name=collection_name, return_properties=app_display_fields, limit=50
        )

        # 3. Rerank search results using semantic reranker
        reranked_results = reranker.rerank(
            results, llm_rewrite_query_text, apply_sigmoid=False, top_k=4
        )

        # 4. Validate token threshold
        valid_results = validate_token_threshold(
            reranked_results,
            llm_rewrite_query_text,
            huberman_system_message,
            tokenizer=encoding,
            token_threshold=6000,
            llm_verbosity_level=llm_verbsoity
        )

        # 5. Generate context series
        context_series = generate_prompt_series(llm_rewrite_query_text, valid_results, llm_verbsoity)
        
        # 6. Add messages to conversations
        st.session_state[UI_CONVERSATION_KEY].add_message(
            Message(role="user", content=user_input)
        )

        # 7. Generate assistant response
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

        # 8. Add assistant response to the conversation
        st.session_state[UI_CONVERSATION_KEY].add_message(
            Message(role="assistant", content=gpt_answer)
        )

        # 9. Add the completed query to the context queue
        completed_query = CompletedQuery(
            user_query=user_input,
            llm_answer=gpt_answer,
            llm_revised_query=llm_rewrite_query_text if llm_rewrite_query_type != "Original" else None
        )
        st.session_state[LLM_CONTEXT_QUEUE].add_query(completed_query)


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
