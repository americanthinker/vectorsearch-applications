from typing import Literal

huberman_system_message = '''
You are a repository of knowledge and wisdom about the Huberman Lab podcast.
The Huberman Lab podcast is hosted by Dr. Andrew Huberman, a neuroscientist and tenured professor of neurobiology 
and ophthalmology at Stanford School of Medicine. The podcast discusses neuroscience and science-based tools, 
including how our brain and its connections with the organs of our body control our perceptions, our behaviors, 
and our health, as well as existing and emerging tools for measuring and changing how our nervous system works. 
The podcast is frequently ranked in the top 10 of all podcasts globally and is often ranked #1 in the categories 
of Science, Education, and Health & Fitness.
Your task is to answer questions about the Huberman Lab podcast only using the context provided in the user message.
Only use the context provided to answer the question. Do not use any external knowledge or resources to answer the question.
'''

question_answering_prompt_single = '''
Use the below context enclosed in triple back ticks to answer the question. If the context does not provide enough information to answer the question, then use any knowledge you have to answer the question.\n
```{context}```\n
Question:\n
{question}.\n
Answer: 
'''

question_answering_user_message = '''
Your task is to synthesize and reason over a series of transcripts of an interview between Andrew Huberman and his guest(s).
After your synthesis, use the series of transcripts to answer the below question.  The series will be in the following format:\n
```
Show Summary: <summary>
Show Guest: <guest>
Transcript: <transcript>
```\n
Start Series:
```
{series}
```
Question:\n
{question}\n
------------------------
1. If the context does not provide enough information to answer the question, then
state that you cannot answer the question with the provided context.
2. Do not use any external knowledge or resources to answer the question.
3. Answer the question directly and {verbosity}.
------------------------
Answer:
'''
verbosity_options = {0: 'concisely', 
                     1: 'use about four sentences', 
                     2: 'with as much detail as possible, within the limits of the context'
    }

context_block = '''
Show Summary: {summary}
Show Guest: {guest}
Transcript: {transcript}
------------------------
'''


def create_context_blocks(results: list[dict],
                          summary_key: str='summary',
                          guest_key: str='guest',
                          content_key: str='content'
                          ) -> list[str]:
    '''
    Creates a list of context blocks (strings) from a list of dictionaries 
    containing the summary, guest, and transcript content.
    '''
    context_series = [context_block.format(summary=res[summary_key],
                                           guest=res[guest_key],
                                           transcript=res[content_key]) 
                      for res in results]
    return context_series

def generate_prompt_series(query: str, 
                           results: list[dict], 
                           verbosity_level: Literal[0, 1, 2]=0,
                           summary_key: str='summary',
                           guest_key: str='guest',
                           content_key: str='content',
                           base_user_prompt: str=question_answering_user_message
                           ) -> str:
    """
    Generates a prompt for an LLM by joining the context blocks of the top results.
    Provides context to the LLM by supplying the summary, guest, and retrieved content of each result.

    Args:
    -----
        query : str
            User query
        results : list[dict]
            List of results from the Weaviate client
    """
    verbosity_levels = [0, 1, 2]
    if not isinstance(verbosity_level, int) or verbosity_level not in verbosity_levels:
        raise ValueError(f'verbosity_level must be an integer: {verbosity_levels}')
    verbosity = verbosity_options[verbosity_level]
    context_series = f'\n'.join(create_context_blocks(results, summary_key, guest_key, content_key)).strip()
    prompt = base_user_prompt.format(question=query, series=context_series, verbosity=verbosity)
    return prompt