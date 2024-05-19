import os
import uuid
from typing import List
from uuid import uuid4

import anthropic
from anthropic.types import ContentBlockDeltaEvent
from dotenv import load_dotenv
from langmem import Client
from langsmith import traceable

load_dotenv()
os.environ["LANGMEM_API_URL"] = os.environ.get("LANGMEM_API_URL")
os.environ["LANGMEM_API_KEY"] = os.environ.get("LANGMEM_API_KEY")
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"langmen - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGSMITH_API_KEY')
os.environ["api_key"] = os.environ.get('anthropic_key')

client = Client()
anthropic_client = anthropic.Anthropic(api_key=os.environ.get('anthropic_key'))


class MyMemoryBot:
    def __init__(self, thread_id: str, user_id: str):
        self.thread_id = thread_id
        self.user_id = user_id

    @traceable(name="Claude", run_type="llm")
    def chat(self, messages: list, model: str = "claude-3-haiku-20240307"):
        system_prompt = messages[0]["content"]
        messages = messages[1:]
        response = anthropic_client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=1024,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            yield chunk

    def get_messages(self):
        """Fetch this thread's messages from LangMem."""
        messages = client.list_messages(self.thread_id)
        print("--------------------")
        print("therad messages ", messages)

        res = []
        for m in messages:
            print(m)
            print("--------------------")
            res.append(m)
        return res

    def strip_metadata(self, messages):
        return [
            {k: v for k, v in m.items() if k not in {"name", "metadata"}}
            for m in messages
        ]

    @traceable
    def invoke_model(self, messages: list):
        # Invoke the model and yield just the text.
        chunks = self.chat(messages)
        for chunk in chunks:
            if isinstance(chunk, ContentBlockDeltaEvent):
                yield chunk.delta.text

    @traceable(run_type="prompt")
    def format_prompt(self, messages: List[dict], user_id: str, thread_id: str):
        new_query = messages[-1]["content"]
        system_prompt = "You're a helpful AI assistant. Be an inquisitive and personable friend to them. Get to know " \
                        "them well! "
        system_prompt += self.query_memory(new_query, user_id)
        messages = [{"role": "system", "content": system_prompt}, *messages]
        return {"messages": messages}

    # Fetch user memories
    def query_memory(self, query: str, user_id: str):
        user_profile = ""
        if user_id:
            mem_result = client.query_user_memory(user_id, text=query)
            memories = mem_result["memories"]
            if memories:
                formatted = "\n".join([mem["text"] for mem in memories])
                print("*************************")
                print("formatted: ", formatted)
                print("*************************")
                user_profile = f"""
    Below are memories from past interactions:
    {formatted}

    End of memories.
    """
        # print("formatted prompt :", formatted)
        return user_profile

    @traceable
    def invoke_model(self, messages: list):
        # Invoke the model and yield just the text.
        chunks = self.chat(messages)
        for chunk in chunks:
            if isinstance(chunk, ContentBlockDeltaEvent):
                yield chunk.delta.text

    def memchat(self, text: str):
        messages = self.get_messages()
        messages.append(
            {"role": "user", "content": text, "metadata": {"user_id": self.user_id}}
        )
        prompt = self.format_prompt(
            self.strip_metadata(messages), self.user_id, self.thread_id
        )
        # print("prompt ", prompt)
        stream = self.invoke_model(**prompt)
        response = ""
        for tok in stream:
            # print(tok, end="")
            response += tok
        # print("response is ", response)
        messages.append({"role": "assistant", "content": response})
        # Post the messages to LangMem
        client.add_messages(self.thread_id, messages=messages[-2:])

        return response


def trigger_mem(thread_id):
    client.trigger_all_for_thread(thread_id)


def main():
    thread_id = str(uuid.uuid4())
    print("thread id: ", thread_id)
    user_id = str(uuid.uuid4())
    print("user id: ", user_id)
    # user_id = "80bda4b3-5236-4063-b59f-98bdde000c18"
    # user_id = "73d3c694-4e05-4478-bf33-fde9ad38e926"
    bot = MyMemoryBot(thread_id, user_id)
    question = input("Hi, what do you want to discuss?\n")
    while True:
        result = bot.memchat(question)
        print("Answer: ", result)
        trigger_mem(thread_id)
        question = input("\n")
        if question == "exit":
            exit()

if __name__ == "__main__":
    main()
