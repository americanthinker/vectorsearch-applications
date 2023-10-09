import os
import openai
from typing import List, Any
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
_ = load_dotenv('./.env', override=True) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

class GPT_Turbo:

    def __init__(self, model: str="gpt-3.5-turbo"):
        self.model = model

    def get_completion_from_messages(self, 
                                     prompt: str, 
                                     temperature: int=0, 
                                     max_tokens: int=500,
                                     show_response: bool=False
                                     ) -> str:
        message = {'role': 'user',
                   'content': prompt}
        
        response = openai.ChatCompletion.create(
                                                model=self.model,
                                                messages=[message],
                                                temperature=temperature,
                                                max_tokens=max_tokens)
        if show_response:
            return response
        return response.choices[0].message["content"]
    
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
    
    def write_to_file(self, file_handle, data: str) -> None:
            file_handle.write(data)
            file_handle.write('\n')
