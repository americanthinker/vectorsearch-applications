import os
from openai import OpenAI
from typing import List, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
_ = load_dotenv('./.env', override=True) # read local .env file


class GPT_Turbo:

    def __init__(self, model: str="gpt-3.5-turbo-0613", api_key: str=os.environ['OPENAI_API_KEY']):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def get_chat_completion(self, 
                            prompt: str, 
                            system_message: str='You are a helpful assistant.',
                            temperature: int=0, 
                            max_tokens: int=500,
                            stream: bool=False,
                            show_response: bool=False
                            ) -> str:
        messages =  [
            {'role': 'system', 'content': system_message},
            {'role': 'assistant', 'content': prompt}
                    ]
        
        response = self.client.chat.completions.create( model=self.model,
                                                        messages=messages,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        stream=stream)
        if show_response:
            return response
        return response.choices[0].message.content
    
    def multi_thread_request(self, 
                             filepath: str,
                             prompt: str,
                             content: List[str],
                             temperature: int=0
                             ) -> List[Any]:
        
        data = []
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as exec:
            futures = [exec.submit(self.get_completion_from_messages, [{'role': 'user','content': f'{prompt} ```{c}```'}], temperature, 500, False) for c in content]
            with open(filepath, 'a') as f:
                for future in as_completed(futures):
                    result = future.result()
                    if len(data) % 10 == 0:
                            print(f'{len(data)} of {len(content)} completed.')
                    if result:
                        data.append(result)
                        self.write_to_file(file_handle=f, data=result)
        return [res for res in data if res]
    
    def generate_question_context_pairs(self, 
                                        context_tuple: Tuple[str, str], 
                                        num_questions_per_chunk: int=2, 
                                        max_words_per_question: int=10
                                        ) -> List[str]:
        
        doc_id, context = context_tuple
        prompt = f'Context information is included below enclosed in triple backticks. Given the context information and not prior knowledge, generate questions based on the below query.\n\nYou are an end user querying for information about your favorite podcast. \
                   Your task is to setup {num_questions_per_chunk} questions that can be answered using only the given context. The questions should be diverse in nature across the document and be no longer than {max_words_per_question} words. \
                   Restrict the questions to the context information provided.\n\
                   ```{context}```\n\n'
        
        response = self.get_completion_from_messages(prompt=prompt, temperature=0, max_tokens=500, show_response=True)
        questions = response.choices[0].message["content"]
        return (doc_id, questions)

    def batch_generate_question_context_pairs(self,
                                              context_tuple_list: List[Tuple[str, str]],
                                              num_questions_per_chunk: int=2,
                                              max_words_per_question: int=10
                                              ) -> List[Tuple[str, str]]:
        data = []
        progress = tqdm(unit="Generated Questions", total=len(context_tuple_list))
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as exec:
            futures = [exec.submit(self.generate_question_context_pairs, context_tuple, num_questions_per_chunk, max_words_per_question) for context_tuple in context_tuple_list]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    data.append(result)
                    progress.update(1)
        return data
    
    def get_embedding(self):
         pass
    
    def write_to_file(self, file_handle, data: str) -> None:
            file_handle.write(data)
            file_handle.write('\n')
