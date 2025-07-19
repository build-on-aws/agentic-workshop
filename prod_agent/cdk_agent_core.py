from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient
import json
import logging
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from contextlib import ExitStack

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the AgentCore App
app = BedrockAgentCoreApp()

# Initialize Bedrock model
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    temperature=0.7,
)

# Define system prompt
SYSTEM_PROMPT = """
You are an expert AWS Certified Solutions Architect. Your role is to help customers understand best practices on building on AWS. You have the following capabilities:

1. Query AWS Documentation: You can search and retrieve information from AWS documentation to provide accurate and up-to-date guidance.
2. Understand AWS CDK: You can explain AWS Cloud Development Kit (CDK) concepts, patterns, and best practices for infrastructure as code.
3. Retrieve AWS Pricing Information: You can look up and explain pricing details for AWS services to help customers understand cost implications.

When answering questions:
- Provide concise, accurate information based on official AWS documentation
- Include relevant code examples when explaining CDK concepts
- Suggest cost-effective architectural solutions
- Always cite your sources when referencing specific AWS documentation
"""

# Global variables to maintain MCP client instances and context
mcp_clients = None
exit_stack = None
agent_instance = None

def create_mcp_clients():
    """Create and return MCP client instances"""
    aws_docs_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"]
            )
        )
    )

    aws_cdk_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx", args=["awslabs.cdk-mcp-server@latest"]
            )
        )
    )

    aws_pricing_client = MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx", args=["awslabs.aws-pricing-mcp-server@latest"]
            )
        )
    )

    
    return aws_docs_client, aws_cdk_client, aws_pricing_client

def initialize_agent():
    """
    Initialize the agent with MCP tools.
    Ensures MCP clients are properly started and maintained.
    """
    global mcp_clients, exit_stack, agent_instance
    
    # If we already have an initialized agent, return it
    if agent_instance is not None:
        return agent_instance
    
    logger.info("Initializing MCP clients and agent...")
    
    # Create a new exit stack to manage context managers
    exit_stack = ExitStack()
    
    # Create MCP clients
    aws_docs_client, aws_cdk_client, aws_pricing_client = create_mcp_clients()
    mcp_clients = [aws_docs_client, aws_cdk_client, aws_pricing_client]
    
    # Enter all MCP client contexts using the exit stack
    for client in mcp_clients:
        exit_stack.enter_context(client)
    
    # Get all tools from the MCP clients
    all_tools = aws_docs_client.list_tools_sync() + aws_cdk_client.list_tools_sync() + aws_pricing_client.list_tools_sync()
    
    # Create the agent with the tools
    agent_instance = Agent(tools=all_tools, model=bedrock_model, system_prompt=SYSTEM_PROMPT)
    
    logger.info("Agent initialized successfully with MCP tools")
    return agent_instance

def cleanup_resources():
    """Clean up MCP client resources"""
    global exit_stack, agent_instance, mcp_clients
    
    if exit_stack is not None:
        logger.info("Cleaning up MCP client resources...")
        exit_stack.close()
        exit_stack = None
        agent_instance = None
        mcp_clients = None
        logger.info("Resources cleaned up")

# Define the entrypoint for AgentCore Runtime
@app.entrypoint
def docs_diag_agent(payload):
    """
    Invoke the agent with a payload
    
    Args:
        payload (dict): Contains the prompt from the user
        
    Returns:
        str: The agent's response text
    """
    try:
        user_input = payload.get("prompt")
        logger.info(f"User input: {user_input}")
        
        # Initialize agent with MCP tools (will reuse existing if already initialized)
        agent = initialize_agent()
        
        # Get response from agent
        response = agent(user_input)
        
        # Extract and return the text content from the response
        return response.message['content'][0]['text']
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        # Clean up resources on error to allow for fresh initialization on next request
        cleanup_resources()
        raise

# Run the app when executed directly
if __name__ == "__main__":
    try:
        app.run()
    finally:
        # Ensure resources are cleaned up when the app exits
        cleanup_resources()