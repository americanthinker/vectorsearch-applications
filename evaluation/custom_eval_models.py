from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models.gpt_model import GPTModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from anthropic import Anthropic, AsyncAnthropic
from openai import AzureOpenAI, AsyncAzureOpenAI

from dataclasses import dataclass
from typing import Literal
import os


class CustomAnthropic(DeepEvalBaseLLM):
    """
    Creates a custom evaluation model interface that uses the Anthropic API 
    for evaluation of metrics. 
    """
    def __init__(
                self,
                model: Literal['claude-3-haiku-20240307', 'claude-3-sonnet-2024022', 'claude-3-opus-20240229']
                ):
        self.accepted_model_types = ['claude-3-haiku-20240307', 'claude-3-sonnet-2024022', 'claude-3-opus-20240229']
        if model not in self.accepted_model_types:
            raise ValueError(f'{model} is not found in {self.accepted_model_types}. Retry with acceptable model type')
        self.model = model

    def load_model(self, async_mode: bool=False) -> AsyncAnthropic | Anthropic:
        if async_mode:
            return AsyncAnthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        return Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        message = client.messages.create(
                                        max_tokens=1024,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": prompt,
                                            }
                                        ],
                                        model=self.model,
                                        )
        if message:
            return message.content[0].text
        return "no message returned"

    async def a_generate(self, prompt: str) -> str:
        aclient = self.load_model(async_mode=True)
        message = await aclient.messages.create(
                                                max_tokens=1024,
                                                messages=[
                                                    {
                                                        "role": "user",
                                                        "content": prompt,
                                                    }
                                                ],
                                                model=self.model,
                                                )
        if message:
            return message.content[0].text
        return "no message returned"

    def get_model_name(self) -> str:
        return "Custom Anthropic Model"
    

class CustomAzureOpenAI(GPTModel):
    def __init__(
                self,
                deployment_name: str
                ) -> None:
        self.model = deployment_name

    def load_model(self, async_mode: bool=False) -> AzureOpenAI | AsyncAzureOpenAI:
        if async_mode:
            return AsyncAzureOpenAI(azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                                    api_key=os.environ['AZURE_OPENAI_API_KEY'],
                                    api_version=os.environ['AZURE_OPENAI_API_VERSION'],
                                    )
        return AzureOpenAI(azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                           api_key=os.environ['AZURE_OPENAI_API_KEY'],
                           api_version=os.environ['AZURE_OPENAI_API_VERSION'],
                           )

    def generate(self, prompt: str) -> str:
        client = self.load_model()
        completion = client.chat.completions.create(model=self.model,
                                                    messages=[
                                                            {
                                                        "role": "user",
                                                        "content": prompt,
                                                            }
                                                        ],
                                                    max_tokens=1024)
        if completion:
            return completion.choices[0].message.content
        return "no message returned"

    async def a_generate(self, prompt: str) -> str:
        aclient = self.load_model(async_mode=True)
        completion = await aclient.chat.completions.create(model=self.model,
                                                            messages=[
                                                                {
                                                                    "role": "user",
                                                                    "content": prompt,
                                                                }
                                                            ],
                                                            max_tokens=1024)
        if completion:
            return completion.choices[0].message.content
        return "no message returned"

    def get_model_name(self) -> str:
        return "Custom Azure OpenAI Model"
    

class AnswerCorrectnessMetric(GEval):
    '''
    Custom metric to determine correctness of an LLM generated answer
    as related to the retrieval context. Takes in LLM model string as 
    single param. 
    '''
    name = 'AnswerCorrectness'
    evaluation_steps=['Compare the actual output with the retrieval context to verify factual accuracy.',
        'Assess if the actual output effectively addresses the specific information requirement stated in the input.',
        'Determine the comprehensiveness of the actual output in addressing all key aspects mentioned in the input.',
        'Score the actual output between 0 and 1, based on the accuracy and comprehensiveness of the information provided.',
        'If there is not enough information in the retrieval context to correctly answer the input, and the actual output indicates that the input cannot be answered with the provided context, then the score should be 1.']
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]

    def __init__(self, model: str | DeepEvalBaseLLM) -> None:
        self.model = model
        super().__init__(name=self.name,
                         evaluation_steps=self.evaluation_steps,
                         model=self.model,
                         evaluation_params=self.evaluation_params)
    
@dataclass
class EvalResponse:
    score: float
    reason: str
    metric: str
    cost: float
    eval_model: str
    eval_steps: list[str]
    input: str = None
    actual_output: str = None
    retrieval_context: list[str] = None

@dataclass
class TestCaseBundle:
    inputs: list[str]
    actual_outputs: list[str]
    retrieval_contexts: list[list[str]]
    test_cases: list[LLMTestCase]