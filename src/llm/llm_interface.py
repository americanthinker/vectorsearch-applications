from litellm import completion, acompletion
from litellm.utils import CustomStreamWrapper, ModelResponse
import os

class LLM:
    '''
    Creates primary Class instance for interacting with various LLM model APIs.
    Primary APIs supported are OpenAI and Anthropic.
    '''
    # non-exhaustive list of supported models 
    # these models were tested and are known to work with the LLM class (April 2024)
    valid_models = {'openai': [
                        "gpt-4-turbo-preview",
                        "gpt-4-0125-preview",
                        "gpt-4-1106-preview",
                        "gpt-3.5-turbo",
                        "gpt-3.5-turbo-1106",
                        "gpt-3.5-turbo-0125",
                        ],
                    'anthropic': [ 'claude-3-haiku-20240307', 
                                   'claude-3-sonnet-2024022',
                                   'claude-3-opus-20240229'
                                   ],
                    'cohere': ['command-r',
                               'command-r-plus'
                               ]
                    }

    def __init__(self, 
                 model_name: str='gpt-3.5-turbo-0125',
                 api_key: str=None,
                 api_version: str=None,
                 api_base: str=None
                 ):
        self.model_name = model_name
        if not api_key:
            try:
                self._api_key = os.environ['OPENAI_API_KEY']
            except KeyError:
                raise ValueError('Default api_key expects OPENAI_API_KEY environment variable. Check that you have this variable or pass in another api_key.')
        else:
            self._api_key = api_key
        self.api_version = api_version
        self.api_base = api_base
        if self.api_base and self.api_version:
            #if both base and version are present, assume user is using Azure OpenAI API
            self.model_name = f"azure/{self.model_name}"

  
    def chat_completion(self, 
                        system_message: str,
                        user_message: str='',
                        temperature: int=0, 
                        max_tokens: int=500,
                        stream: bool=False,
                        raw_response: bool=False,
                        **kwargs
                        ) -> str | CustomStreamWrapper | ModelResponse:
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
        if raw_response or stream:
            return response
        return response.choices[0].message.content
    
    async def achat_completion(self, 
                               system_message: str,
                               user_message: str=None,
                               temperature: int=0, 
                               max_tokens: int=500,
                               stream: bool=False,
                               raw_response: bool=False,
                               **kwargs
                               ) -> str | CustomStreamWrapper | ModelResponse:
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
        if raw_response or stream:
            return response
        return response.choices[0].message.content
