import openai
from reranker import ReRanker
import tiktoken
from openai_interface import GPT_Turbo
from opensearch_interface import OpenSearchClient
from retriever_pipeline import retrieve_pipeline, generate_prompt
from prompt_templates import question_answering_system, question_answering_prompt
import streamlit as st
import torch
from streamlit_pills import pills
from dotenv import load_dotenv
load_dotenv('./.env', override=True)
import os

openai.api_key = os.environ['OPENAI_API_KEY']
gpt = GPT_Turbo()
osclient = OpenSearchClient()
reranker = ReRanker()
tokenizer = tiktoken.get_encoding('cl100k_base')
st.subheader("AI Assistant : Streamlit + OpenAI: `stream` *argument*")

# You can also use radio buttons instead
selected = pills("", ["NO Streaming", "Streaming"], ["🎈", "🌈"])

query = st.text_input("You: ",placeholder = "Ask me anything ...", key="input")

response = retrieve_pipeline(query=query,
                             index_name='impact-theory-minilm-196',
                             search_type='hybrid',
                             retriever=osclient,
                             reranker=reranker,
                             tokenizer=tokenizer,
                             kw_size=50,
                             vec_size=50,
                             top_k=5,
                             return_text=False)

prompt = generate_prompt(base_prompt=question_answering_prompt, query=query, results=response)

if st.button("Submit", type="primary"):
    st.markdown("----")
    res_box = st.empty()
    
    if selected == "Streaming":
        report = []
        # Looping over the response
        for resp in gpt.get_completion_from_messages(
                                                    prompt=prompt,
                                                    system_message=question_answering_system,
                                                    max_tokens=250, 
                                                    stream = True,
                                                    show_response=True):
            # join method to concatenate the elements of the list 
            # into a single string, 
            # then strip out any empty strings
            try:
                report.append(resp.choices[0].delta.get('content', '\n'))
                result = "".join(report).strip()
                result = result.replace("\n", "")    
                res_box.markdown(f'*{result}*') 
            except Exception as e:
                print(e)
                continue
        for i, resp in enumerate(response, 1):
            title = resp['_source']['title']
            st.text_area(label=title, value=resp['_source']['content'], height=250)
            st.write('---')
    else:
        completions = openai.Completion.create(model='text-davinci-003',
                                            prompt=prompt,
                                            max_tokens=120, 
                                            temperature = 0.5,
                                            stream = False)
        result = completions.choices[0].text
        
        res_box.write(result)
st.markdown("----")
torch.cuda.empty_cache()
