# from ibm_watsonx_ai import APIClient
# from ibm_watsonx_ai import Credentials

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
# from langchain_ibm import ChatWatsonx
from langchain_huggingface import ChatHuggingFace

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from typing import Annotated
from typing_extensions import TypedDict

import json


with open("apikey.json", "r") as f:
    _apikey = json.load(f)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str


tavi_key: str = _apikey["tavily"]
# apikey: str = _apikey["apikey"]
# project_id: str = _apikey["project_id"]
# credentials: Credentials = Credentials(
#                                 url = "https://us-south.ml.cloud.ibm.com",
#                                 api_key = apikey,
#                             )
# client: APIClient = APIClient(credentials)


@tool
def human_assistance(
        name: str, 
        birthday: str, 
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)

# Replace TavilySearchResults with your own search package (Homemade selenium, scrapy, requests... anything that works for free)
search_tool = TavilySearchResults(
    max_results=2,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=tavi_key
        )
    )
# tools = [search_tool, human_assistance]
tools = [search_tool]

llm = ChatHuggingFace(
                model_name="meta-llama/llama-3-70b-instruct",
                model_kwargs={"temperature": 0.1}
            )
# llm = ChatWatsonx(
#                 model_id="meta-llama/llama-3-3-70b-instruct",
#                 watsonx_client=client,
#                 project_id=project_id
#             )
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
tool_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    prompt = {"messages": [{"role": "user", "content": user_input}]}

    for event in graph.stream(
            prompt,
            config,
            stream_mode="values"
        ):
        event["messages"][-1].pretty_print()


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break