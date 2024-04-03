from litellm import completion_with_retries, completion, acompletion
import pandas as pd
import os

class LLM:

    def __init__(self, 
                 model_name: str="gpt-3.5-turbo-0125", 
                 api_key: str=os.environ['OPENAI_API_KEY']
                 ):
        self.model_name = model_name

    def get_chat_completion(self, 
                            system_message: str,
                            assistant_message: str=None,
                            temperature: int=0, 
                            max_tokens: int=500,
                            stream: bool=False,
                            raw_response: bool=False,
                            **kwargs
                            ) -> str:
        
        initial_role = 'user' if self.model_name.startswith('claude') else 'system'
        if self.model_name.startswith('claude'):
            temperature = temperature/2
        messages =  [
            {'role': initial_role, 'content': system_message},
            {'role': 'assistant', 'content': assistant_message}
                    ]
        
        response = completion_with_retries(model=self.model_name,
                                           messages=messages,
                                           temperature=temperature,
                                           max_tokens=max_tokens,
                                           stream=stream,
                                           retry_strategy="exponential_backoff_retry",
                                           **kwargs)
        if raw_response:
            return response
        return response.choices[0].message.content
    
    
    # def generate_question_context_pairs(self, 
    #                                     context_tuple: Tuple[str, str], 
    #                                     num_questions_per_chunk: int=2, 
    #                                     max_words_per_question: int=10
    #                                     ) -> List[str]:
        
    #     doc_id, context = context_tuple
    #     prompt = f'Context information is included below enclosed in triple backticks. Given the context information and not prior knowledge, generate questions based on the below query.\n\nYou are an end user querying for information about your favorite podcast. \
    #                Your task is to setup {num_questions_per_chunk} questions that can be answered using only the given context. The questions should be diverse in nature across the document and be no longer than {max_words_per_question} words. \
    #                Restrict the questions to the context information provided.\n\
    #                ```{context}```\n\n'
        
    #     response = self.get_completion_from_messages(prompt=prompt, temperature=0, max_tokens=500, show_response=True)
    #     questions = response.choices[0].message["content"]
    #     return (doc_id, questions)

    # def batch_generate_question_context_pairs(self,
    #                                           context_tuple_list: List[Tuple[str, str]],
    #                                           num_questions_per_chunk: int=2,
    #                                           max_words_per_question: int=10
    #                                           ) -> List[Tuple[str, str]]:
    #     data = []
    #     progress = tqdm(unit="Generated Questions", total=len(context_tuple_list))
    #     with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as exec:
    #         futures = [exec.submit(self.generate_question_context_pairs, context_tuple, num_questions_per_chunk, max_words_per_question) for context_tuple in context_tuple_list]
    #         for future in as_completed(futures):
    #             result = future.result()
    #             if result:
    #                 data.append(result)
    #                 progress.update(1)
    #     return data
    
    # def get_embedding(self):
    #      pass
    
    # def write_to_file(self, file_handle, data: str) -> None:
    #         file_handle.write(data)
    #         file_handle.write('\n')
