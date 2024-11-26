from litellm import completion, acompletion
from litellm.utils import CustomStreamWrapper, ModelResponse
import os

class LLM:
    '''
    Creates primary Class instance for interacting with various LLM model APIs.
    Primary APIs supported are OpenAI and Anthropic.
    '''
    # non-exhaustive list of supported models 
    # these models were tested and are known to work with the LLM class (November 2024)
    valid_models = {'openai': [
                        "gpt-4o",
                        "gpt-4o-mini"
                        ],
                    'anthropic': [ 'claude-3-sonnet-20240229', 
                                   'claude-3-haiku-20240307'
                                   ],
                    'cohere': ['command-r',
                               'command-r-plus'
                               ]
                    }

    def __init__(self, 
                 model_name: str='gpt-4o-mini',
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

    def _handle_response(self,
                         response: CustomStreamWrapper | ModelResponse, 
                         raw_response: bool
                         ) -> str | CustomStreamWrapper | ModelResponse:
        """Handles response formatting into a string or returns original raw response."""
        if raw_response:
            return response
        try:
            return response.choices[0].message.content
        except Exception as e:
            print(f'Error: {e}')
            return response
    
    def _create_message_block(self, 
                              system_message: str, 
                              user_message: str
                              ) -> list[dict[str, str]]:
        """Creates a message block for the model."""
        messages =  [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
                    ]
        return messages
    
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
        messages = self._create_message_block(system_message, user_message)
        response = completion(model=self.model_name,
                              messages=messages,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              stream=stream,
                              api_key=self._api_key,
                              api_base=self.api_base,
                              api_version=self.api_version,
                              **kwargs)
        return self._handle_response(response, raw_response)
    
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

        messages =  self._create_message_block(system_message, user_message)
        response = await acompletion(model=self.model_name,
                                     messages=messages,
                                     temperature=temperature,
                                     max_tokens=max_tokens,
                                     stream=stream,
                                     api_key=self._api_key,
                                     api_base=self.api_base,
                                     api_version=self.api_version,
                                     **kwargs)
        return self._handle_response(response, raw_response)
