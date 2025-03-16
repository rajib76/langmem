import os
from typing import ClassVar, Tuple, List
from uuid import uuid4

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel

from memory_stores.sqlite_store import SQLLITESTORE
load_dotenv()
unique_id = uuid4().hex[0:8]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = f"newlangmen - {unique_id}"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = os.environ.get('LANGSMITH_API_KEY')
index = {
    "dims": 1536,
    "embed": "openai:text-embedding-3-small"
}


class ChatAgentWithSemanticMemory(BaseModel):
    store: ClassVar = SQLLITESTORE(db_path="sqllite.sqlite", index_config=index)
    namespace: Tuple = ("agent_memories", "{user_id}")
    checkpointer: ClassVar = InMemorySaver()
    memory_tools: List = [
        create_manage_memory_tool(namespace),
        create_search_memory_tool(namespace)
    ]

    def get_chat_agent(self):
        agent = create_react_agent("openai:gpt-4o", tools=self.memory_tools, store=self.store,
                                   checkpointer=self.checkpointer)

        return agent


if __name__ == "__main__":
    chat_agent = ChatAgentWithSemanticMemory()
    agent = chat_agent.get_chat_agent()

    while True:
        user_query = input("Ask your query")
        if "exit" in user_query:
            exit()
        thread_id = "thread-1"
        user_id = "User-A"
        result_state = agent.invoke({"messages": [{"role": "user", "content": user_query}]},
                                    config={"configurable": {"thread_id": thread_id, "user_id": user_id}})

        print(result_state["messages"][-1].content)
