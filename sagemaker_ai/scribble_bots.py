import os
from textwrap import dedent

import litellm
from dotenv import load_dotenv

load_dotenv()

from crewai import LLM, Agent, Crew, Process, Task

os.environ["LITELLM_LOG"] = "DEBUG"
litellm.set_verbose = True

llm = LLM(
    model="sagemaker_chat/jumpstart-dft-deepseek-llm-r1-disti-20250207-153847",
    temperature=0.7,
    max_tokens=100,
)


research_agent = Agent(
    role="Research Bot",
    goal="Scan sources, extract relevant information, and compile a research summary.",
    backstory="An AI agent skilled in finding relevant information from a variety of sources.",
    allow_delegation=True,
    llm=llm,
    verbose=False,
)

writer_agent = Agent(
    role="Writer Bot",
    goal="Receive research summaries and transform them into structured content.",
    backstory="A talented writer bot capable of producing high-quality, structured content based on research.",
    allow_delegation=False,
    llm=llm,
    verbose=False,
)

research_task = Task(
    description=(
        "Your task is to conduct research based on the following query: {prompt}.\n"
        "- Scan multiple sources to gather relevant information.\n"
        "- Summarize your findings into a concise, well-organized research summary."
    ),
    expected_output="A comprehensive research summary based on the provided query.",
    agent=research_agent,
)

writing_task = Task(
    description=(
        "Your task is to create structured content based on the research provided.\n"
        "- Transform the research into high-quality written content.\n"
        "- Ensure the output is clear, organized, and adheres to professional writing standards.\n"
        "- Focus on maintaining the key points while improving readability and flow."
    ),
    expected_output="A well-structured article based on the research summary.",
    agent=writer_agent,
)


scribble_bots = Crew(
    agents=[research_agent, writer_agent],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # Ensure tasks execute in sequence
)


def format_response(raw_text):
    """Format the raw response text for better readability"""
    formatted_text = raw_text.replace("\\n", "\n").strip()
    return formatted_text


def parse_agent_response(response):
    """Helper function to parse and structure agent responses"""
    thought = "Could not extract Chain of Thought."
    answer = "Could not extract Final Answer."

    try:
        if isinstance(response, str):
            response = format_response(response)

            if "Thought:" in response:
                thought_start = response.find("Thought:") + len("Thought:")
                thought_end = (
                    response.find("'Final Answer':")
                    if "'Final Answer':" in response
                    else len(response)
                )
                thought = response[thought_start:thought_end].strip()

            if "'Final Answer':" in response:
                answer_start = response.find("'Final Answer':") + len("'Final Answer':")
                answer = response[answer_start:].strip()

    except Exception as e:
        print(f"Error parsing response: {e}")

    return thought, answer


# Execute the workflow
print("Starting ScribbleBots Sequential Workflow...")
result = scribble_bots.kickoff(
    inputs={"prompt": "Explain what is deepseek very briefly"}
)

print(result)
