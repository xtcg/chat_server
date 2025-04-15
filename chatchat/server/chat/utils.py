import logging
from functools import lru_cache
from typing import Dict, List, Tuple, Union

from langchain.prompts.chat import ChatMessagePromptTemplate

from chatchat.server.pydantic_v2 import BaseModel, Field
from chatchat.utils import build_logger


logger = build_logger()


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """

    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role == "assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw:
        # 将整个内容当作常量字符串，不进行变量替换
            return ChatMessagePromptTemplate.from_template(
                template=self.content,
                role=role,
                input_variables=[] 
            )
        else:
            # 正常模板语法，变量由调用 ChatPromptTemplate 时传入
            return ChatMessagePromptTemplate.from_template(
                template=self.content,
                role=role
            )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h
