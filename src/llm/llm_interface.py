from litellm import completion, acompletion, completion_cost
from typing import Literal
import os

class LLM:
    '''
    Creates primary Class instance for interacting with various LLM model APIs.
    Primary APIs supported are OpenAI and Anthropic.
    '''
    def __init__(self, 
                 model_name: Literal['gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
                                     'claude-3-haiku-20240307', 'claude-3-sonnet-2024022','claude-3-opus-20240229'],
                 api_key: str=os.environ['OPENAI_API_KEY'],
                 api_version: str=None,
                 api_base: str=None
                 ):
        self.model_name = model_name
        self._api_key = api_key
        self.api_version = api_version
        self.api_base = api_base
        self.valid_openai_models = [
                                    "gpt-4-turbo-preview",
                                    "gpt-4-0125-preview",
                                    "gpt-4-1106-preview",
                                    "gpt-4",
                                    "gpt-4-32k",
                                    "gpt-4-0613",
                                    "gpt-4-32k-0613",
                                    "gpt-3.5-turbo",
                                    "gpt-3.5-turbo-1106",
                                    "gpt-3.5-turbo-16k",
                                    "gpt-3.5-turbo-0125",
                                   ]

    def chat_completion(self, 
                        system_message: str,
                        user_message: str='',
                        temperature: int=0, 
                        max_tokens: int=500,
                        stream: bool=False,
                        raw_response: bool=False,
                        return_cost: bool=False,
                        **kwargs
                        ) -> str:
        '''
        Generative text completion method.

        Args:
        -----
        system_message: str
            The system message to be sent to the model.
        user_message: str
            The user message to be sent to the model.
        temperature: int
            The temperature parameter for the model.
        max_tokens: int
            The maximum tokens to be generated.
        stream: bool
            Whether to stream the response.
        raw_response: bool
            If True, returns the raw model response.
        '''
        #reformat roles for claude models
        initial_role = 'user' if self.model_name.startswith('claude') else 'system'
        secondary_role = 'assistant' if self.model_name.startswith('claude') else 'user'
        #handle temperature for claude models
        if self.model_name.startswith('claude'):
            temperature = temperature/2

        messages =  [
            {'role': initial_role, 'content': system_message},
            {'role': secondary_role, 'content': user_message}
                    ]
        
        response = completion(model=self.model_name,
                              messages=messages,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              stream=stream,
                              api_key=self._api_key,
                              api_base=self.api_base,
                              api_version=self.api_version,
                              **kwargs)
        cost = completion_cost(response, model=self.model_name, messages=messages, call_type='completion')
        if raw_response:
            return response
        content = response.choices[0].message.content
        if return_cost:
            return content, cost
        return content
    
    async def achat_completion(self, 
                               system_message: str,
                               user_message: str=None,
                               temperature: int=0, 
                               max_tokens: int=500,
                               stream: bool=False,
                               raw_response: bool=False,
                               return_cost: bool=False,
                               **kwargs
                               ) -> str:
        '''
        Asynchronous generative text completion method.

        Args:
        -----
        system_message: str
            The system message to be sent to the model.
        user_message: str
            The user message to be sent to the model.
        temperature: int
            The temperature parameter for the model.
        max_tokens: int
            The maximum tokens to be generated.
        stream: bool
            Whether to stream the response.
        raw_response: bool
            If True, returns the raw model response.
        '''
        initial_role = 'user' if self.model_name.startswith('claude') else 'system'
        if self.model_name.startswith('claude'):
            temperature = temperature/2
        messages =  [
            {'role': initial_role, 'content': system_message},
            {'role': 'user', 'content': user_message}
                    ]
        response = await acompletion(model=self.model_name,
                                     messages=messages,
                                     temperature=temperature,
                                     max_tokens=max_tokens,
                                     stream=stream,
                                     api_key=self._api_key,
                                     api_base=self.api_base,
                                     api_version=self.api_version,
                                     **kwargs)
        cost = completion_cost(response, model=self.model_name, messages=messages,call_type='completion')
        if raw_response:
            return response
        content = response.choices[0].message.content
        if return_cost:
            return content, cost
        return content
