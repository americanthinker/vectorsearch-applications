from pydantic import BaseModel, Field

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

qa_flavors = ['be highly semantically related. If I were to measure the cosine similarity of their respective vector embeddings, the outcome would be close to 1.0.', 
              'not be semantically related at all. If I were to measure the cosine similarity of their respective vector embeddings, the outcome would be close to 0.',
              'not contain any keyword overlap between the two texts. A comparison of the two texts would show that they share no keywords.',
              'not contain any keyword overlap between the two texts. A comparison of the two texts would show that they share no keywords.']

dataset_generation_system_prompt = '''
Your primary task in life is to generate questions and answers solely about the Huberman Lab podcast
The Huberman Lab podcast is hosted by Dr. Andrew Huberman, a neuroscientist and tenured professor of neurobiology 
and ophthalmology at Stanford School of Medicine. The podcast discusses neuroscience and science-based tools, 
including how our brain and its connections with the organs of our body control our perceptions, our behaviors, 
and our health, as well as existing and emerging tools for measuring and changing how our nervous system works. 
The podcast is frequently ranked in the top 10 of all podcasts globally and is often ranked #1 in the categories 
of Science, Education, and Health & Fitness.

Your task is to create questions and answers to those questions about the Huberman Lab podcast only using the context provided in the user message.
Only use the context provided to answer the question. Do not use any external knowledge or resources to answer the question.
'''

dataset_generation_user_prompt2 = '''
Transcript chunk from a Huberman Lab podcast episode is below. Use the following snippet of episode transcript \
and not prior knowledge, generate a question that can be answered by the transcript section: 

---------------------
Transcript: {transcript}
---------------------

1. Your task is to create an interesting and thought-provoking question that can be answered with the information provided in the transcript.
Follow these question-generation rules explicitly:\n
    1. The question should start with Why, How, or What, and be a question that you would find interesting to discuss with others. The preference is for questions that start with Why or How. 
    2. Do not make any reference to the transcript or episode when generating the question(s), simply generate the question. \
       In other words, do not include the words "transcript", "excerpt", or "episode", in the generated question.
    3. The question generated must contain an answer found within the transcript.
    4. There should not be any keyword overlap between the two query and the transcript. A comparison of the two texts would show that they share no keywords.
    5. If you cannot create a question that meets the above criteria, simply state that you cannot generate a question.
2. Once you have generated a question, provide the answer to the question. 
   1. The answer should be concise and answer the question directly without exposition. 
   2. Return your output in the following format:
    {{
        "question": "Why is the sky blue?",
        "answer": "The sky is blue because of Rayleigh scattering."
    }}

'''

class QAGenerationResponse(BaseModel):
    question: str = Field(description='The question generaated from the transcript provided.  The question should start with Why, How, or What, and be a question that you would find interesting to discuss with others.')
    answer: str = Field(description='The answer to the question generated from the transcript provided.  The answer should be concise and answer the question directly without exposition.')

class QAValidationResponse(BaseModel):
    validation: int = Field(description='This is a binary response field, only acceptable answers are either a "1" or a "0".')
    reasoning: str = Field(description='The reasoning behind the validation response.  This should provide insight into why the question is or is not answerable by the text and why the answer is or is not correct.')

qa_triplet_generation_prompt = '''
You will be provided with a snippet of the Huberman Lab podcast transcript and the guest on the show.
Your task is to follow the below instructions explicitly:\n
    1. Generate a question that can only be answered by the information found in the transcript. The question generated and the transcript should be highly semantically related. This is a Positive example.\n 
    2. Do not make any reference to the transcript or episode when generating the question, simply generate the question.\n
    3. Generate another question that is similar to the original question that you created, but cannot be answered by the information in the transcript. This new question and the original question should be semantically similar.  This is a Hard Negative example.\n
    4. The questions should randomly start with How, Why, or What.\n   
    5. Return your answer in JSON format with the following structure:\n
        {{
            "positive": "How is the process of increasing muscle hypertrophy described?",
            "hard_negative": "How is the process of increasing bone density described?"
        }}

Example: 
---------------------
Guest: Andy Galpin
---------------------
Transcript: "Another academic who's superb in this whole space of muscle physiology, and from a lengthy conversation that I had with Andy, Dr. Galpin, prior to this episode. So if we want to think about muscle hypertrophy, we have to ask what is changing when muscles get larger or stronger? And there are really just three ways that muscles can be stimulated to change. So let's review those three ways and talk about what happens inside the muscle. So there are three major stimuli for changing the way that muscle works and making muscles stronger, larger, or better in some way. And those are stress, tension, and damage. Those three things don't necessarily all have to be present, but stress of some kind has to exist."
---------------------
Response: 
{{
    "positive": "What is required for muscle to change?", 
    "hard_negative": "What is required to do well in the sport of bodybuilding?"
}}


Task
---------------------
Guest: {guest}
---------------------
Transcript: {transcript}
---------------------
Response: 

'''
qa_validation_system_prompt = '''
Your primary goal in life is to provide Quality Assurance of the questions and answers generated about the Huberman Lab podcast.
The Huberman Lab podcast is hosted by Dr. Andrew Huberman, a neuroscientist and tenured professor of neurobiology
and ophthalmology at Stanford School of Medicine. The podcast discusses neuroscience and science-based tools,
including how our brain and its connections with the organs of our body control our perceptions, our behaviors,
and our health, as well as existing and emerging tools for measuring and changing how our nervous system works.
The podcast is frequently ranked in the top 10 of all podcasts globally and is often ranked #1 in the categories
of Science, Education, and Health & Fitness.

Your task is to validate the questions and answers generated about the Huberman Lab podcast. You will be provided with a snippet of the Huberman Lab podcast 
transcript and a question generated from the transcript along with an answer to the question. You will ensure that the question can be answered by the text and that the answer provided is correct.
'''

qa_validation_user_prompt = '''
The following is a snippet of the Huberman Lab podcast along with a question generated from the transcript and an answer to the question: 
---------------------
Transcript: {transcript}
Question: {question}
Answer: {answer}
---------------------
Your task is as follows:
    1. Given the information provided in the transcript, determine if the question is answerable by the text. If the question can be answered by the text then proceed to step 2, otherwise respond with a 0.
    2. Given that the question is answerable by the text, determine if the answer provided is correct. If the answer is correct, respond with a 1. If the answer is incorrect, respond with a 0.
Your output should be in the following JSON format:
{{
    "validation": 1,
    "reasoning": "The question can be answered by the text and the answer provided is correct because it correctly states the main facts."
}}
'''