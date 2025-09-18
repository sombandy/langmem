from typing import Annotated

from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.cache import SQLiteCache
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.utils.config import get_store
from langmem import ReflectionExecutor, create_memory_store_manager

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

memory_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("memories",),
)

executor = ReflectionExecutor(memory_manager, store=get_store())
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

tools = [YahooFinanceNewsTool()]
llm = init_chat_model("openai:gpt-5-mini")
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list[AnyMessage], add_messages]

async def chat(state: State):
    messages_with_memories = await prompt(state)
    response = await llm_with_tools.ainvoke(messages_with_memories)

    last_user = state["messages"][-1]
    user_content = getattr(last_user, "content", last_user[1] if isinstance(last_user, tuple) else str(last_user))
    to_process = {"messages": [{"role": "user", "content": user_content}, response]}

    delay = 15  # Typically 30-60 minutes in production
    executor.submit(to_process, after_seconds=delay)

    return {"messages": [response]}


tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chat", chat)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chat", tools_condition)
graph_builder.add_edge("tools", "chat")
graph_builder.add_edge(START, "chat")
agent = graph_builder.compile()
