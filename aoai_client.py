from azure_aad import get_or_refresh_token
from openai import AzureOpenAI, AsyncAzureOpenAI

class AOAI:
    def __init__(self, 
                 model: str='gpt-3.5-turbo-0125',
                 azure_endpoint: str="https://smt-openai-agg-gpt4-eastus2.openai.azure.com/", 
                 api_version: str='2024-02-01', 
                 asynch: bool=False
                 ):
        self.model = model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.asynch = asynch

        azure_ad_token = get_or_refresh_token()
        if asynch:
            self.client = AsyncAzureOpenAI(azure_endpoint=azure_endpoint, 
                                           api_version=api_version, 
                                           azure_ad_token=azure_ad_token)
        else:
            self.client = AzureOpenAI(azure_endpoint=azure_endpoint, 
                                      api_version=api_version, 
                                      azure_ad_token=azure_ad_token)
        
    def query(self, 
              user_message: str,
              system_message: str='You are an extremely bright and capable assistant.  Fearless in bringing the truth to higher power.'
              ) -> dict:
        messages = [{'role': 'system', 'content':system_message},
                    {'role': 'user', 'content': user_message}]
        response = self.client.chat.completions.create(messages=messages, model=self.model)
        if response:
            return response.choices[0].message.content
        
    async def aquery(self, 
                     user_message: str,
                     system_message: str='You are an extremely bright and capable assistant.  Fearless in bringing the truth to higher power.'
                     ) -> dict:
        if not self.asynch:
            raise ValueError('This method is only available for asynchronous Azure OpenAI clients.')
        messages = [{'role': 'system', 'content':system_message},
                    {'role': 'user', 'content': user_message}]
        response = await self.client.chat.completions.create(messages=messages, model=self.model)
        if response:
            return response.choices[0].message.content
