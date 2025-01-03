import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock


def initialize_settings():
    """
    Initialize global settings for LlamaIndex.
    This sets up the language model (LLM) and embedding model using Amazon Bedrock.
    """
    # Set the LLM to use Haiku model from Bedrock
    Settings.llm = Bedrock(
        model="anthropic.claude-3-5-haiku-20241022-v1:0",
        region_name="us-west-2",
        context_size=2000,
    )
    # Set the embedding model to use Amazon's Titan model
    Settings.embed_model = BedrockEmbedding(
        model="amazon.titan-embed-text-v2:0",
        region_name="us-west-2",
    )


def load_or_create_index(file_path, persist_dir):
    """
    Load an existing index from storage or create a new one if it doesn't exist.

    Args:
    file_path (str): Path to the PDF file to index.
    persist_dir (str): Directory to persist the index.

    Returns:
    VectorStoreIndex: The loaded or newly created index.
    """
    if os.path.exists(persist_dir):
        print(f"Loading existing index from {persist_dir}")
        # Load the existing index from the specified directory
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)
    else:
        print(f"Creating new index from {file_path}")
        # Load documents from the PDF file
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        # Create a new index from the documents
        index = VectorStoreIndex.from_documents(documents)
        # Persist the index to the specified directory
        index.storage_context.persist(persist_dir)
        return index


def create_query_engine_tool(query_engine, name, description):
    """
    Create a QueryEngineTool for use with the ReActAgent.

    Args:
    query_engine: The query engine to use.
    name (str): Name of the tool.
    description (str): Description of the tool.

    Returns:
    QueryEngineTool: A tool that can be used by the ReActAgent.
    """
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name=name, description=description),
    )


def main():
    """
    Main function to orchestrate the index creation/loading and querying process.
    """
    # Initialize LlamaIndex settings
    initialize_settings()

    # Load or create indexes for Lyft and Uber data
    lyft_index = load_or_create_index("./data/10k/lyft_2021.pdf", "./data/lyft_index")
    uber_index = load_or_create_index("./data/10k/uber_2021.pdf", "./data/uber_index")

    # Create query engines from the indexes
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)

    # Create query engine tools for the ReActAgent
    query_engine_tools = [
        create_query_engine_tool(
            lyft_engine,
            "lyft_10k",
            "Provides information about Lyft financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
        create_query_engine_tool(
            uber_engine,
            "uber_10k",
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool.",
        ),
    ]

    # Create a ReActAgent with the query engine tools
    agent = ReActAgent.from_tools(query_engine_tools, verbose=True)

    # Use the agent to answer a question
    response = agent.chat("Compare revenue growth of Uber and Lyft from 2020 to 2021")
    print(response)


if __name__ == "__main__":
    main()
