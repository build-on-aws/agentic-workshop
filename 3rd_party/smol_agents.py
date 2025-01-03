from huggingface_hub import list_models
from smolagents import CodeAgent, LiteLLMModel, ManagedAgent
from transformers import tool


@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which
    """
    most_downloaded_model = next(
        iter(list_models(filter=task, sort="downloads", direction=-1))
    )
    return most_downloaded_model.id


def multi_agent_example():
    """
    This is an example of how to use the CodeAgent to create a multi-agent system.
    """
    llm = LiteLLMModel(model_id="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0")

    model_download_agent = CodeAgent(tools=[model_download_tool], model=llm)

    managed_model_download_agent = ManagedAgent(
        agent=model_download_agent,
        name="model_download",
        description="Returns the most downloaded model of a given task on the Hugging Face Hub.",
    )

    manager_agent = CodeAgent(
        tools=[],
        model=llm,
        managed_agents=[managed_model_download_agent],
    )

    manager_agent.run(
        "Find the most downloaded model of text generation on the Hugging Face Hub."
    )


multi_agent_example()
