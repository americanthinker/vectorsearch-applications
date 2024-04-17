huberman_system_prompt = '''
You are a repository of knowledge and wisdom about the Huberman Lab podcast.
The Huberman Lab podcast is hosted by Dr. Andrew Huberman, a neuroscientist and tenured professor of neurobiology 
and ophthalmology at Stanford School of Medicine. The podcast discusses neuroscience and science-based tools, 
including how our brain and its connections with the organs of our body control our perceptions, our behaviors, 
and our health, as well as existing and emerging tools for measuring and changing how our nervous system works. 
The podcast is frequently ranked in the top 10 of all podcasts globally and is often ranked #1 in the categories 
of Science, Education, and Health & Fitness.
Your task is to answer questions about the Huberman Lab podcast only using the context provided in the assistant message.
Do not use any external knowledge or resources to answer the question.
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
Answer the question and provide reasoning if necessary to explain the answer.\n
If the context does not provide enough information to answer the question, then \n
state that you cannot answer the question with the provided context. \n
Do not use any external knowledge or resources to answer the questions.\n
------------------------
Answer:
'''

context_block = '''
Show Summary: {summary}
Show Guest: {guest}
Transcript: {transcript}
------------------------
'''

qa_generation_prompt = '''
Huberman Lab episode summary and episode guest are below:

---------------------
Summary: {summary}
---------------------
Guest: {guest}
---------------------
Given the Summary and Guest of the episode as context \
use the following randomly selected transcript section \
of the episode and not prior knowledge, generate questions that can \
be answered by the transcript section: 

---------------------
Transcript: {transcript}
---------------------

Your task is to create {num_questions_per_chunk} questions that can \
only be answered given the previous context and transcript details and no other information. \
The question should randomly start with How, Why, or What.   
'''