"""Small input/output models shared by prompts."""

from pydantic import BaseModel


class StringIO(BaseModel):
    text: str

    def __hash__(self):
        return hash(self.text)


class BoolIO(BaseModel):
    value: bool

    def __hash__(self):
        return hash(self.value)
