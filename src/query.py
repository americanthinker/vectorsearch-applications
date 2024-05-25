from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Dict, Tuple


@dataclass
class CompletedQuery:
    user_query: str
    llm_answer: str
    llm_revised_query: str = None


class CompletedQueryQueue:

    def __init__(self, max_length: int = 5):
        self.completed_query_list = deque(maxlen=max_length)

    def add_query(self, query: CompletedQuery):
        self.completed_query_list.append(query)

    def format_completed_query_queue(self) -> str:
        formatted_queries = []
        for i, query in enumerate(self.completed_query_list, 1):
            formatted_query = (
                f"Query {i}:\n"
                f"User Query: {query.llm_revised_query if query.llm_revised_query else query.user_query}\n"
                f"LLM Answer: {query.llm_answer}\n"
            )
            formatted_queries.append(formatted_query)
        return "\n\n".join(formatted_queries)
    
    def generate_prompt(self, current_query: str) -> str:
        instructions = (
            "Instructions:\n"
            "This task involves analyzing user queries within the context of their previous interactions to determine their completeness and specificity.\n\n"
            "1. Identify Ambiguity:\n"
            "   - A query is considered vague or ambiguous if:\n"
            "     - It relies on pronouns without clear antecedents (e.g., 'he,' 'she,' 'they').\n"
            "     - It uses non-specific nouns without mentioning specific titles, dates, or identifiable subjects.\n"
            "     - It makes indirect references to topics or subjects previously mentioned without specifying them in the current context.\n\n"
            "2. Rewrite Ambiguous Queries:\n"
            "   - Directly link to specific episodes, events, or topics hinted at but not explicitly specified.\n"
            "   - Use proper nouns and clear descriptors.\n"
            "   - Convert implicit references into explicit questions.\n"
            "   - Replace general terms with specific details.\n\n"
            "3. Examples:\n"
            "   - Original: 'What happened after that?'\n"
            "     Revised: 'What were the main topics discussed in the episode following [Specific Episode Title]?'\n"
            "   - Original: 'What topics are covered?'\n"
            "     Revised: 'What are the key points Dr. X discussed about Y in [Specific Episode]?'\n"
            "   - Original: 'What else did he accomplish in his career?'\n"
            "     Revised: 'What else did George Washington accomplish during his time as president?'\n\n"
            "4. Output Format:\n"
            "   - If the query is clear and unambiguous, respond with: 'Original query: [original query]'\n"
            "   - If the query is ambiguous, respond with: 'Revised query: [revised query]'\n\n"
            "The goal is to ensure each rewritten query is a clear, direct question that unambiguously asks for information about specific topics, episodes, or guests. "
            "This facilitates more accurate and relevant searches or responses, enhancing user satisfaction by providing contextually appropriate replies. This active intervention "
            "is crucial for not only responding accurately but also for anticipating user needs based on prior queries, tailoring each query to continue the thread of discussion seamlessly, "
            "emphasizing clarity and detail.\n"
        )
        
        formatted_context = self.format_completed_query_queue()
        
        prompt = (
            f"{instructions}\n"
            f"Previous Queries and Context:\n"
            f"{formatted_context}\n\n"
            f"Current Query: {current_query}\n\n"
        )
        return prompt

def parse_llm_response(response: str) -> str:
    """
    Parse the response from the LLM to determine if it was an original or revised query and extract the query text.
    
    :param response: The response string from the LLM.
    :return: A tuple containing the type of query ('Original' or 'Revised') and the query text.
    """
    if response.startswith('Original query:'):
        query_text = response[len('Original query: '):].strip()
    elif response.startswith('Revised query:'):
        query_text = response[len('Revised query: '):].strip()
    else:
        raise ValueError("Invalid response format")
    
    return query_text

query_rewrite_system_message = '''
You are a repository of knowledge and wisdom about the Huberman Lab podcast.
The Huberman Lab podcast is hosted by Dr. Andrew Huberman, a neuroscientist and tenured professor of neurobiology 
and ophthalmology at Stanford School of Medicine. The podcast discusses neuroscience and science-based tools, 
including how our brain and its connections with the organs of our body control our perceptions, our behaviors, 
and our health, as well as existing and emerging tools for measuring and changing how our nervous system works. 
The podcast is frequently ranked in the top 10 of all podcasts globally and is often ranked #1 in the categories 
of Science, Education, and Health & Fitness.
Your task is to rewrite ambiguous or vague user queries to make them more specific and clear, so that users 
can receive accurate and relevant responses from the Huberman Lab podcast database.
'''