import time
from typing import List

def convert_seconds(seconds: int):
    """
    Converts seconds to a string of format Hours:Minutes:Seconds
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def generate_prompt(base_prompt: str, query: str, results: List[dict]) -> str:
    """
    Generates a prompt for the OpenAI API
    """
    contexts = '\n\n'.join([r['content'] for r in results])
    prompt = base_prompt.format(question=query, context=contexts)
    return prompt