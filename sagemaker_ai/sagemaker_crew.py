from textwrap import dedent

from dotenv import load_dotenv

load_dotenv()

from crewai import LLM, Agent, Crew, Task

llm = LLM(
    model="sagemaker_chat/jumpstart-dft-deepseek-llm-r1-disti-20250207-153847",
    temperature=0.7,
    max_tokens=100,
)


class BestJobAgents:
    def job_researcher_agent(self):
        return Agent(
            role="Job Researcher",
            goal="Find the single best job opportunity in the given location",
            backstory=dedent(
                """You analyze local job markets to identify the most promising 
                career opportunity in terms of salary, growth, and demand."""
            ),
            allow_delegation=False,
            verbose=True,
            llm=llm,
            max_execution_time=30,
        )

    def content_writer_agent(self):
        return Agent(
            role="Job Writer",
            goal="Create a concise description of the best job opportunity",
            backstory=dedent(
                """You specialize in writing clear, concise job descriptions 
                that highlight key information for job seekers."""
            ),
            allow_delegation=False,
            verbose=True,
            llm=llm,
            max_execution_time=30,
        )

    def editor_agent(self):
        return Agent(
            role="Content Editor",
            goal="Ensure job description is accurate and actionable",
            backstory=dedent(
                """You polish job descriptions to ensure they are accurate, 
                clear, and valuable for job seekers."""
            ),
            allow_delegation=True,
            verbose=True,
            llm=llm,
            max_execution_time=30,
        )


class BestJobTasks:
    def research_task(self, agent, location):
        return Task(
            description=dedent(
                f"""Find the single best job opportunity in {location}. Include:
                1. Job title
                2. Key responsibilities
                3. Required skills
                4. Potential salary"""
            ),
            agent=agent,
            expected_output="Brief summary of best job opportunity",
        )

    def write_job_task(self, agent, location):
        return Task(
            description=dedent(
                f"""Write a concise description of the best job in {location}.
                Keep it under 50 words and cover:
                - Role overview
                - Key requirements
                - Growth potential"""
            ),
            agent=agent,
            expected_output="Concise job description",
        )

    def edit_job_task(self, agent, location):
        return Task(
            description=dedent(
                f"""Polish the job description for {location}. Ensure:
                1. Clarity and accuracy
                2. Professional tone
                3. Actionable insights
                Keep the final text under 75 words."""
            ),
            agent=agent,
            expected_output="Polished job description",
        )


tasks = BestJobTasks()
agents = BestJobAgents()

print("## Welcome to the Best Job Finder")
print("--------------------------------")
location = input("Where would you like to find the best job opportunity?\n")

# Create Agents
job_researcher = agents.job_researcher_agent()
content_writer = agents.content_writer_agent()
editor = agents.editor_agent()

# Create Tasks
research_job = tasks.research_task(job_researcher, location)
write_job = tasks.write_job_task(content_writer, location)
edit_job = tasks.edit_job_task(editor, location)

# Create Crew
crew = Crew(
    agents=[job_researcher, content_writer, editor],
    tasks=[research_job, write_job, edit_job],
    verbose=True,
)

result = crew.kickoff()

# Print results
print("\n\n########################")
print("## Here is the result")
print("########################\n")
print(f"Best Job Opportunity in {location}:")
print(result)
