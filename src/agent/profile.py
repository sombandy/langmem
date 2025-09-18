import asyncio

from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.globals import set_llm_cache
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import create_manage_memory_tool

load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

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

tools = [
    YahooFinanceNewsTool(),
    create_manage_memory_tool(namespace=("memories",)),
]
agent = create_react_agent("openai:gpt-4o-mini", prompt=prompt, tools=tools)


async def test_agent():
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )
    agent.store = store

    async for chunk in agent.astream(
        {"messages": [("user", "I'm Som, I have purchased some tesla stocks")]},
        stream_mode="updates"
    ):
        print(chunk)
        print("\n")

    print(store.search(("memories",)))

if __name__ == "__main__":
    asyncio.run(test_agent())
