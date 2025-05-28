from datetime import datetime

from mcp import StdioServerParameters, stdio_client
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

aws_docs_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"]
        )
    )
)

aws_diag_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx", args=["awslabs.aws-diagram-mcp-server@latest"]
        )
    )
)

# Cost Analysis MCP Client
cost_analysis_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx", args=["awslabs.cost-analysis-mcp-server@latest"]
        )
    )
)

# PowerPoint MCP Client
ppt_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["--from", "office-powerpoint-mcp-server", "ppt_mcp_server"],
        )
    )
)


bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    # model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
    temperature=0.7,
)

COST_ANALYSIS_AGENT_PROMPT = """
You are a cost analysis specialist with expertise in:
- Analyzing AWS cost structures and pricing models
- Performing detailed cost projections and optimization recommendations
- Creating cost comparison scenarios for migration planning
- Identifying cost-saving opportunities across AWS services
- Building cost monitoring and alerting strategies
- Analyzing Reserved Instance and Savings Plan opportunities
- Providing detailed cost breakdowns by service, region, and usage patterns
Use the cost analysis tools to provide accurate financial projections and optimization strategies.
"""

SA_AGENT_PROMPT = """
You are an AWS Certified Solutions Architect with expertise in:
- Creating detailed architecture diagrams using AWS services
- Performing comprehensive cost analysis and optimization
- Writing technical documentation and runbooks
- Analyzing security and compliance requirements
- Designing for high availability, fault tolerance, and disaster recovery
Use the AWS documentation and diagram tools to create accurate, professional deliverables.
"""


@tool
def cost_analysis_specialist(query: str) -> str:
    """
    Analyze costs and create financial projections for migration.
    This tool agent specializes in AWS cost analysis and optimization strategies.
    """
    with aws_docs_client, cost_analysis_client:
        all_tools = (
            aws_docs_client.list_tools_sync() + cost_analysis_client.list_tools_sync()
        )
        cost_agent = Agent(
            system_prompt=COST_ANALYSIS_AGENT_PROMPT,
            tools=all_tools,
            model=bedrock_model,
        )
        return str(cost_agent(query))


@tool
def presentation_creator(query: str) -> str:
    """
    Create executive presentations with PowerPoint.
    This tool agent specializes in creating professional presentations.
    """
    with ppt_client:
        ppt_agent = Agent(
            system_prompt="""You create professional PowerPoint presentations for executive audiences.
            Focus on clear visualizations, key metrics, and strategic recommendations.
            Use charts, diagrams, and bullet points effectively.""",
            tools=ppt_client.list_tools_sync(),
            model=bedrock_model,
        )
        return str(ppt_agent(query))


@tool
def architecture_analyst(query: str) -> str:
    """
    Create architecture diagrams and perform cost analysis.
    This tool agent specializes in AWS architecture design and cost optimization.
    """
    with aws_docs_client, aws_diag_client:
        all_tools = (
            aws_docs_client.list_tools_sync() + aws_diag_client.list_tools_sync()
        )
        sa_agent = Agent(
            system_prompt=SA_AGENT_PROMPT, tools=all_tools, model=bedrock_model
        )
        response = sa_agent(query)
        # Extract diagram path if created
        if "diagram" in str(response).lower():
            return f"{response}\n\nNote: Check the output for the diagram file path."
        return str(response)


def create_migration_orchestrator():
    """
    Create the main orchestrator agent for cloud migration planning.
    This orchestrator coordinates all specialized tool agents to deliver
    a comprehensive migration plan.
    """

    MIGRATION_ORCHESTRATOR_PROMPT = """
    You are a Cloud Migration Coordinator orchestrating a comprehensive migration plan to AWS.
    
    Your role is to:
    1. Analyze the migration requirements
    2. Delegate specific tasks to specialized agents
    3. Synthesize outputs into a cohesive migration strategy
    
    Available tool agents:
    - architecture_analyst: For diagrams and architectural design
    - cost_analysis_specialist: For cost analysis and financial projections
    - presentation_creator: For executive presentations
    
    For migration projects, follow this process:
    1. Use architecture_analyst to create diagrams and architectural documentation
    2. Use cost_analysis_specialist to analyze costs and create financial projections
    3. Use presentation_creator to build an executive presentation
    
    Always ensure:
    - Security best practices are followed
    - Cost optimization is considered
    - High availability and disaster recovery are planned
    - Final migration plan consolidates all technical and financial analysis
    """

    orchestrator = Agent(
        system_prompt=MIGRATION_ORCHESTRATOR_PROMPT,
        tools=[
            architecture_analyst,
            cost_analysis_specialist,
            presentation_creator,
        ],
        model=bedrock_model,
    )

    return orchestrator


def run_cloud_migration_demo():
    """
    Demo: Cloud Migration Planning using Agents as Tools pattern

    This demonstrates how an orchestrator agent delegates to specialized
    tool agents to create a comprehensive migration plan.
    """
    print("=" * 70)
    print("Cloud Migration Planning (Agents as Tools Pattern)")
    print("=" * 70)
    print("\nPattern: Hierarchical delegation with specialized tool agents")
    print("Scenario: Planning cloud migration for e-commerce company\n")

    # Track execution start time
    start_time = datetime.now()

    # Create the orchestrator
    orchestrator = create_migration_orchestrator()

    # Define the migration request
    migration_request = """
    Plan a comprehensive cloud migration for "ShopEasy" e-commerce company:
    
    Current State:
    - On-premise monolithic Java application
    - MySQL database with 50TB of data
    - 1 million daily active users
    - Peak traffic during sales events (10x normal)
    - Legacy file storage system with 100TB of product images
    
    Requirements:
    - Zero downtime migration
    - High availability across multiple regions
    - Cost optimization (current spend: $100K/month)
    - Compliance with PCI-DSS for payment processing
    - Improved performance and scalability
    
    Constraints:
    - 6-month migration timeline
    - $500K migration budget
    - Limited DevOps expertise in current team
    - Must maintain integration with existing ERP system
    
    Deliverables needed:
    1. Architecture diagrams with migration phases
    2. Detailed cost analysis and projections
    3. Migration runbook and documentation
    4. Executive presentation for board approval
    """

    print("📋 Processing migration request...")
    print("🤖 Orchestrator delegating to specialized agents...\n")

    # Execute the orchestrated migration planning
    result = orchestrator(migration_request)

    # Track execution end time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    print(f"Execution time: {execution_time:.2f} seconds")

    print("\n✅ Migration Plan Generated!")
    print("-" * 70)
    print(result)
    print("-" * 70)


run_cloud_migration_demo()
