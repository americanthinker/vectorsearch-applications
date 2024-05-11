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