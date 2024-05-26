import sqlite3

from src.llm.llm_interface import LLM
from loguru import logger


def get_cursor(file: str = "./sqlite3.db"):
    """
    Creates a sqlite connection and cursor for query execution,
    from a sqlite filepath.
    """
    conn = sqlite3.connect(file)
    cursor = conn.cursor()
    return cursor, conn


class Text2SQL:
    """
    Example pipeline that incorporates, Text2SQL call,
    and a regular hybrid search.
    """

    def __init__(
        self,
        llm: LLM,
        sqldb_path: str = "./sqlite3.db",
    ):
        self.llm = llm
        self.sqldb_path = sqldb_path

    def text2sql(self, query: str):
        """
        Executes text2sql call to LLM.  Input is natural language query.
        Output is the input transformed into a usable SQL command.
        """
        system_message = """
                         Your main function is to generate flawless SQL queries from plain text language, that will enable a user 
                         to query a SQLite3 database.  There is one table in the database called "huberman" with four columns labeled:
                         1. guests
                         2. titles
                         3. summaries
                         4. view_counts
                         """
        base_user_message = """
                            Convert the plain text query into a SQL query that can answer questions about the data contained in the huberman table. 
                            If a specific piece of text information is requested such as a guest name or show title, use the LIKE operator and % wildcard character as appropriate to address the user information need.
                            Return the SQL as a single command, do not include any newlines, and return only the SQL command nothing else.
                            Do not include reasoning or other information, just return the SQL command.
                            --------------
                            PLAIN TEXT QUERY: {query}
                            --------------
                            SQL:
                            """
        user_message = base_user_message.format(query=query)
        return self.llm.chat_completion(system_message, user_message).strip()

    def get_sqldb_response(self, command: str):
        """
        Creates cursor and executes SQL command.
        Closes db connection and returns answer.
        """
        cursor, conn = get_cursor(self.sqldb_path)
        cursor.execute(command)
        answer = cursor.fetchall()
        cursor.close()
        conn.close()
        return answer

    def parse_sql_response(self, query: str, sql_answer: str) -> str:
        """
        Executes single LLM call to translate raw SQL response into a
        human-friendly response.  Requires both original natural language
        query and raw sql response.
        """
        response_system_message = """
                                  You excel at answering questions from a multitude of information sources.  
                                  Specifically you know how to translate SQL-generated answers into actionable insights
                                  """
        base_response_user_message = """
        You will be given a user question and a response to that query pulled from a SQL database.  
        Use the information from the response to answer the original query.  Answer the question in an objective tone, and do not make any reference to the SQL database.
        If the information in the answer is presented in a list, ensure that you return your output in the same order, do not mix up the original ordering. 
        If the response does not answer the question, then either state that you do not know the answer, or answer the question based on prior knowledge.
        ----------------
        USER QUERY: {query}
        ----------------
        ANSWER FROM SQL DATABASE: {sql_answer}
        ----------------
        OUTPUT: 
        """
        response_user_message = base_response_user_message.format(
            query=query, sql_answer=sql_answer
        )
        return self.llm.chat_completion(response_system_message, response_user_message)

    def __call__(self, query: str) -> list[dict] | tuple[list[dict], str]:
        """
        Trigger series of API calls and return gathered context.
        """

        # send text2sql call to LLM
        sql_command = self.text2sql(query)  # llm call
        logger.info(f"SQL Command: {sql_command}")

        # ping SQL database using the sql_command
        sql_answer = self.get_sqldb_response(sql_command)  # sql db call
        logger.info(f"Raw SQL Answer: {sql_answer}")

        # call LLM to parse sql_answer and respond with readable format
        parsed_sql_answer = self.parse_sql_response(query, sql_answer)  # llm call
        logger.info(f"Parsed SQL Answer: {parsed_sql_answer}")

        return parsed_sql_answer
