from pydantic import BaseModel


class Message(BaseModel):
    messageText: str
