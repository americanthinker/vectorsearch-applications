from deepeval.models.base_model import DeepEvalBaseLLM
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
    

class CustomAzureOpenAI(DeepEvalBaseLLM):
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
    

class AnswerCorrectnessMetric:

    def __init__(self, model: str | DeepEvalBaseLLM, strict: bool=True) -> None:
        self.name = "answer_correctness"
        self.model = model
        self.strict = strict

    def get_metric(self) -> GEval:
        return GEval(
            name=self.name,
            criteria="Answer Correctness - Given the information provided in the retrieval context, how accurate \
                     is the actual output and does this output address the information requirement of the input?",
            model=self.model,
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
            strict_mode=self.strict
            )
@dataclass
class EvalResponse:
    metric: str
    model: str
    input: str
    actual_output: str
    retrieval_context: list[str]
    score: float
    reason: str

@dataclass
class TestCaseBundle:
    inputs: list[str]
    actual_outputs: list[str]
    retrieval_contexts: list[list[str]]
    test_cases: list[LLMTestCase]