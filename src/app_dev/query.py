from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Dict

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
    llm_revisied_query: str = None


class CompletedQueryQueue:
    completed_query_list: Deque[CompletedQuery]

    def __init__(self, max_length: int = 5):
        self.completed_query_list = deque(maxlen=max_length)

    def add_query(self, query: CompletedQuery):
        self.completed_query_list.append(query)

    def to_string(self) -> str:
        query_strings = []
        for completed_query in self.completed_query_list:
            # Format each context result into a string
            context_results_str = ', '.join([f"Title: {result.title}, Guest: {result.guest}, Content: {result.content}"
                                             for result in completed_query.context_results_list])
            # Compose the full string for this query
            revised_query_str = f"LLM Revised Query: {completed_query.llm_revisied_query}" if completed_query.llm_revisied_query else "LLM Revised Query: None"
            query_str = (f"User Query: {completed_query.user_query}, "
                         f"Context Results: [{context_results_str}], "
                         f"LLM Answer: {completed_query.llm_answer}, "
                         f"{revised_query_str}")
            query_strings.append(query_str)
        
        # Join all individual query strings into a single string with line breaks between each query description
        return "\n".join(query_strings)
    
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

def create_llm_prompt(new_query, query_history):
    # Preparing the context of previous queries in a structured format
    formatted_history = "Query History:\n"
    for query in query_history.split(', LLM Answer:'):
        if query.strip():
            formatted_history += f"{query.strip()},\n"

    # Creating the prompt with clear instructions for the LLM
    prompt = (
        f"New User Query: \"{new_query}\"\n\n"
        f"{formatted_history}\n"
        "Instructions:\n"
        "This task involves analyzing user queries within the context of their previous interactions to determine their completeness and specificity. "
        "Identify queries as vague or ambiguous if they rely on pronouns without clear antecedents, use non-specific nouns without mentioning "
        "specific titles, dates, or identifiable subjects, or make indirect references to topics or subjects previously mentioned without specifying "
        "them in the current context. When rewriting queries, directly link to specific episodes, events, or topics that are hinted at but not explicitly "
        "specified, using proper nouns and clear descriptors. Convert implicit references into explicit questions, for example, changing "
        "'What happened after that?' to 'What were the main topics discussed in the episode following [Specific Episode Title]?' Replace general terms with "
        "specific details, such as changing 'What topics are covered?' to 'What are the key points Dr. X discussed about Y in [Specific Episode]?' "
        "The goal is to ensure each rewritten query is a clear, direct question that unambiguously asks for information about specific topics, episodes, "
        "or guests. This facilitates more accurate and relevant searches or responses, enhancing user satisfaction by providing contextually appropriate replies. "
        "This active intervention is crucial for not only responding accurately but also for anticipating user needs based on prior queries, tailoring each "
        "query to continue the thread of discussion seamlessly, emphasizing clarity and detail."
    )
    return prompt