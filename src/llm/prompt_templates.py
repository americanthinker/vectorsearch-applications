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

question_answering_prompt_series = '''
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
verbosity_options = ['concisely', 'use about four sentences', 'with as much detail as possible, within the limits of the context']

context_block = '''
Show Summary: {summary}
Show Guest: {guest}
Transcript: {transcript}
------------------------
'''

# ---------------------
# Summary: {summary}

qa_generation_prompt = '''
Huberman Lab episode guest and transcript are below:

---------------------
Guest: {guest}
---------------------
Given the Guest of the episode as context use the following snippet of episode transcript \
and not prior knowledge, generate questions that can be answered by the transcript section: 

---------------------
Transcript: {transcript}
---------------------

Your task is to create {num_questions_per_chunk} questions that can only be answered \
given the transcript content and no other information. Follow these rules explicitly:\n
    1. Do not make any reference to the transcript or episode when generating the question(s), simply generate the question(s).\n
    2. The question generated and the transcript chunk should be highly semantically related.  If I were to measure their respective vector embeddings using cosine similarity, the outcome would be close to 1.0.\n
    3. The question(s) should randomly start with How, Why, or What.   
'''