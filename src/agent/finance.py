from dotenv import load_dotenv

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool

load_dotenv()


async def prompt(state):
    """Prepare the messages for the LLM."""
    store = get_store()
    memories = await store.asearch(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


# store = InMemoryStore(
#     index={
#         "dims": 1536,
#         "embed": "openai:text-embedding-3-small",
#     }
# )
tools = [
    YahooFinanceNewsTool(),
    create_manage_memory_tool(namespace=("memories",)),
]
agent = create_react_agent("openai:gpt-4o-mini", prompt=prompt, tools=tools)

if __name__ == "__main__":
    for chunk in agent.stream(
        {"messages": [("user", "What is the latest news on Tesla?")]},
        stream_mode="updates",
    ):
        print(chunk)
        print("\n")
