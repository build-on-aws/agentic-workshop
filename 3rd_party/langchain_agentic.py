import functools
import operator
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_aws import ChatBedrock
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode


# Define the state object passed between nodes in the graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


# Tool definitions
def setup_tools():
    """Set up and return the tools used by the agents."""
    tavily_tool = TavilySearchResults(max_results=5)

    @tool
    def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."]
    ):
        """Execute Python code and return the result."""
        repl = PythonREPL()
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
        return (
            result_str
            + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )

    return [tavily_tool, python_repl]


# Agent creation
def create_agent(llm, tools, system_message: str):
    """Create an agent with specified LLM, tools, and system message."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants. "
                "Use the provided tools to progress towards answering the question. "
                "If you are unable to fully answer, that's OK, another assistant with different tools "
                "will help where you left off. Execute what you can to make progress. "
                "If you or any of the other assistants have the final answer or deliverable, "
                "prefix your response with FINAL ANSWER so the team knows to stop. "
                "You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(
        system_message=system_message,
        tool_names=", ".join([tool.name for tool in tools]),
    )
    return prompt | llm.bind_tools(tools)


# Node functions
def agent_node(state, agent, name):
    """Process the state through an agent and return the updated state."""
    result = agent.invoke(state)
    if not isinstance(result, ToolMessage):
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }


def setup_workflow(llm, tools):
    """Set up and return the workflow graph."""
    # Create agents
    research_agent = create_agent(
        llm, tools, "You should provide accurate data for the chart_generator to use."
    )
    chart_agent = create_agent(
        llm, tools, "Any charts you display will be visible by the user."
    )

    # Create nodes
    research_node = functools.partial(
        agent_node, agent=research_agent, name="Researcher"
    )
    chart_node = functools.partial(
        agent_node, agent=chart_agent, name="chart_generator"
    )
    tool_node = ToolNode(tools)

    # Set up the workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("Researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_node("call_tool", tool_node)

    # Add edges
    workflow.add_conditional_edges(
        "Researcher",
        router,
        {"continue": "chart_generator", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "chart_generator",
        router,
        {"continue": "Researcher", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
        "call_tool",
        lambda x: x["sender"],
        {"Researcher": "Researcher", "chart_generator": "chart_generator"},
    )
    workflow.add_edge(START, "Researcher")

    return workflow.compile()


# Router function
def router(state) -> Literal["call_tool", "__end__", "continue"]:
    """Determine the next step in the workflow based on the current state."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        return "__end__"
    return "continue"


# Main execution
def main():
    # Set up the LLM
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs=dict(temperature=0),
        region_name="us-west-2",
    )

    # Set up tools
    tools = setup_tools()

    # Set up the workflow
    graph = setup_workflow(llm, tools)

    # Execute the workflow
    events = graph.stream(
        {
            "messages": [
                HumanMessage(
                    content="Fetch the UK's GDP over the past 5 years, "
                    "then create a bar graph for me to see. "
                    "Once you code it up, save the bar graph as a png"
                )
            ],
        },
        {"recursion_limit": 150},
    )

    # Print the results
    for s in events:
        print(s)
        print("----")


if __name__ == "__main__":
    main()
