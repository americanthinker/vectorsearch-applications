"""A module to represent a conversation between a user and an assistant."""

from typing import Deque, List, Dict
from collections import deque
from pydantic import BaseModel, validator
from typing import Literal


# Define the roles as a type that can only have specific values
RoleType = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: RoleType
    content: str

    # Optional: Add a validator to ensure content is not empty
    @validator("content")
    def content_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Content must not be empty")
        return v


class Conversation:
    message_queue: Deque[Message]

    def __init__(self, conversation_id: str, system_message: Message, max_length: int = 100):
        self.conversation_id = conversation_id
        self.message_queue = deque(maxlen=max_length)  # Fixed-size queue
        self.add_message(system_message)

    def add_message(self, message: Message):
        self.message_queue.append(message)

    def add_messages_from_dicts(self, messages_dicts: List[Dict]):
        # Convert a list of dictionaries to Message instances and add them to the queue
        for message_dict in messages_dicts:
            self.add_message(Message(**message_dict))

    def queue_to_list(self) -> List[dict]:
        # Serialize message queue to a list of dictionaries
        return [message.model_dump() for message in self.message_queue]

    def load_messages(self, messages: List[dict]):
        # Directly use add_messages_from_dicts for loading messages from a list of dicts
        self.add_messages_from_dicts(messages)