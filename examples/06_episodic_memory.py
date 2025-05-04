from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager
from pydantic import BaseModel, Field

load_dotenv()


class Episode(BaseModel):
    """Write the episode from the perspective of the agent within it. Use the benefit of hindsight to record the
    memory, saving the agent's key internal thought process so it can learn over time. """

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
                    ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )


# Set up vector store for similarity search
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Configure memory manager with storage
manager = create_memory_store_manager(
    "openai:gpt-4o",
    namespace=("memories", "episodes"),
    schemas=[Episode],
    instructions="Extract exceptional examples of noteworthy problem-solving scenarios, including what made them "
                 "effective.",
    enable_inserts=True,
)

llm = init_chat_model("openai:gpt-4o")


@entrypoint(store=store)
def app(messages: list):
    # Step 1: Find similar past episodes
    print("messages ", messages)
    similar = store.search(
        ("memories", "episodes"),
        query=messages[-1]["content"],
        limit=1,
    )

    # Step 2: Build system message with relevant experience
    system_message = "You are a helpful assistant."
    if similar:
        system_message += "\n\n### EPISODIC MEMORY:"
        for i, item in enumerate(similar, start=1):
            episode = item.value["content"]
            system_message += f"""

Episode {i}:
User ask: {episode['observation']}
Thought: {episode['thoughts']}
Response: {episode['action']}
Result: {episode['result']}
        """

    # Step 3: Generate response using experience
    print("System Prompt ", system_message)
    print("User Question ", messages[-1]["content"])
    response = llm.invoke([{"role": "system", "content": system_message}, *messages])

    # Step 4: Store this interaction if successful
    manager.invoke({"messages": messages})
    return response


conversation = [
    {
        "role": "user",
        "content": "Can you help me get the names of all customers who placed orders in January?",
    },
    {
        "role": "assistant",
        "content": (
            "Sure! Assuming you have a table called `customers` and another called `orders` with a `customer_id` "
            "column linking them, "
            "and `orders` has a `order_date` column, you can use the following SQL:\n\n"
            "```sql\n"
            "SELECT c.name\n"
            "FROM customers c\n"
            "JOIN orders o ON c.id = o.customer_id\n"
            "WHERE MONTH(o.order_date) = 1;\n"
            "```\n"
            "This query gets all customer names who placed orders in January."
        ),
    },
    {
        "role": "user",
        "content": "Thanks! That helps. I’ll try it out.",
    },
]

response = app.invoke(
    conversation
)
print("response ", response)

conversation = [
    {
        "role": "user",
        "content": "Can you help me get the names of all customers who placed orders in February?",
    }]
response = app.invoke(
    conversation
)
print("response ", response.content)
# print(store.search(("memories", "episodes"),
#                    query="Can you help me get the names of all customers who placed orders in Feb"))
