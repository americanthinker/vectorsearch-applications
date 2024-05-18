from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Dict, Tuple

@dataclass
class ContextResult:
    title: str
    guest: str
    content: str


@dataclass
class CompletedQuery:
    user_query: str
    context_results_list: List[ContextResult]
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
            formatted_results = []
            for j, result in enumerate(query.context_results_list, 1):
                formatted_results.append(
                    f"Context Result {j}:\n"
                    f"Title: {result.title}\n"
                    f"Guest: {result.guest}\n"
                    f"Content: {result.content}\n"
                )
            formatted_query = (
                f"Query {i}:\n"
                f"User Query: {query.user_query}\n"
                f"LLM Answer: {query.llm_answer}\n"
                f"LLM Revised Query: {query.llm_revised_query if query.llm_revised_query else 'None'}\n"
                f"{''.join(formatted_results)}"
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
            "   - If the query is clear and unambiguous, respond with: 'Original query: [query]'\n"
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
            f"Determine if the current query is clear or ambiguous. If it is clear, respond with 'Original query: {current_query}'. "
            f"If it is ambiguous, rewrite it for clarity and respond with 'Revised query: [revised query]'."
        )
        
        return prompt
    
def parse_context_results(data: List[Dict[str, any]]) -> Deque[ContextResult]:
    results: Deque[ContextResult] = deque()
    for item in data:
        if 'title' in item and 'guest' in item and 'content' in item:
            # Create a new ContextResult object and append it to the deque
            result = ContextResult(
                title=item['title'],
                guest=item['guest'],
                content=item['content']
            )
            results.append(result)
    return results

def parse_llm_response(response: str) -> Tuple[str, str]:
    """
    Parse the response from the LLM to determine if it was an original or revised query and extract the query text.
    
    :param response: The response string from the LLM.
    :return: A tuple containing the type of query ('Original' or 'Revised') and the query text.
    """
    if response.startswith('Original query:'):
        query_type = 'Original'
        query_text = response[len('Original query: '):].strip()
    elif response.startswith('Revised query:'):
        query_type = 'Revised'
        query_text = response[len('Revised query: '):].strip()
    else:
        raise ValueError("Invalid response format")
    
    return query_type, query_text

# def create_llm_prompt(new_query, query_history):
#     # Preparing the context of previous queries in a structured format
#     formatted_history = "Query History:\n"
#     for query in query_history.split(', LLM Answer:'):
#         if query.strip():
#             formatted_history += f"{query.strip()},\n"

#     # Creating the prompt with clear instructions for the LLM
#     prompt = (
#         f"New User Query: \"{new_query}\"\n\n"
#         f"{formatted_history}\n"
#         "Instructions:\n"
#         "This task involves analyzing user queries within the context of their previous interactions to determine their completeness and specificity. "
#         "Identify queries as vague or ambiguous if they rely on pronouns without clear antecedents, use non-specific nouns without mentioning "
#         "specific titles, dates, or identifiable subjects, or make indirect references to topics or subjects previously mentioned without specifying "
#         "them in the current context. When rewriting queries, directly link to specific episodes, events, or topics that are hinted at but not explicitly "
#         "specified, using proper nouns and clear descriptors. Convert implicit references into explicit questions, for example, changing "
#         "'What happened after that?' to 'What were the main topics discussed in the episode following [Specific Episode Title]?' Replace general terms with "
#         "specific details, such as changing 'What topics are covered?' to 'What are the key points Dr. X discussed about Y in [Specific Episode]?' "
#         "The goal is to ensure each rewritten query is a clear, direct question that unambiguously asks for information about specific topics, episodes, "
#         "or guests. This facilitates more accurate and relevant searches or responses, enhancing user satisfaction by providing contextually appropriate replies. "
#         "This active intervention is crucial for not only responding accurately but also for anticipating user needs based on prior queries, tailoring each "
#         "query to continue the thread of discussion seamlessly, emphasizing clarity and detail. Please only either respond with a revised qeury or the original query"
#         "in the following format:\n\n"
#         "If the query is clear and unambiguous, respond with 'Original query: [query]'\n"
#         "If the query is ambiguous, respond with 'Revised query: [revised query]'\n"
#     )
#     return prompt