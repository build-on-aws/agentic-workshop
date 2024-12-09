import json
import math
import os
import shutil
import subprocess
import zipfile
from typing import List

import boto3
from botocore.exceptions import ClientError

# Retrieve environment variables
LAMBDA_ROLE = os.environ["LAMBDA_ROLE"]
S3_BUCKET = os.environ["S3_BUCKET"]
REGION = "us-west-2"


def initialize_clients():
    """Initialize and return the AWS Bedrock, Lambda, and S3 clients."""
    session = boto3.Session()
    bedrock = session.client(service_name="bedrock-runtime", region_name=REGION)
    lambda_client = session.client("lambda", region_name=REGION)
    s3 = session.client("s3", region_name=REGION)
    return bedrock, lambda_client, s3


def get_working_directory():
    """
    Returns the appropriate working directory based on the environment.
    In AWS Lambda, this will be /tmp. Otherwise, it will be the current directory.
    """
    if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
        # We're likely in AWS Lambda
        return "/tmp"
    return os.getcwd()


def create_deployment_package_no_dependencies(
    lambda_code, project_name, output_zip_name
):
    """
    Create a deployment package without dependencies.
    Works in both local and AWS Lambda environments.
    """
    # Get the appropriate working directory
    working_dir = get_working_directory()

    # Create full paths
    project_path = os.path.join(working_dir, project_name)
    output_zip_path = os.path.join(working_dir, output_zip_name)

    try:
        # Create the project directory
        os.makedirs(project_path, exist_ok=True)

        # Write the lambda code to the lambda_function.py file
        lambda_file_path = os.path.join(project_path, "lambda_function.py")
        with open(lambda_file_path, "w") as f:
            f.write(lambda_code)

        # Create a .zip file for the deployment package
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            zipf.write(lambda_file_path, "lambda_function.py")  # Path inside zip

        return output_zip_path

    finally:
        # Clean up the project directory
        if os.path.exists(project_path):
            shutil.rmtree(project_path)


def create_deployment_package_with_dependencies(
    lambda_code, project_name, output_zip_name, dependencies
):
    """
    Create a deployment package with dependencies.
    Works in both local and AWS Lambda environments.
    """
    # Get the appropriate working directory
    working_dir = get_working_directory()

    # Create full paths
    project_path = os.path.join(working_dir, project_name)
    output_zip_path = os.path.join(working_dir, output_zip_name)
    package_dir = os.path.join(project_path, "package")

    try:
        # Create the project directory
        os.makedirs(project_path, exist_ok=True)

        # Write the lambda code to the lambda_function.py file
        lambda_file_path = os.path.join(project_path, "lambda_function.py")
        with open(lambda_file_path, "w") as f:
            f.write(lambda_code)

        # Install the dependencies to the package directory
        os.makedirs(package_dir, exist_ok=True)

        # Install dependencies
        if isinstance(dependencies, str):
            dependencies = dependencies.split(",")

        for dependency in dependencies:
            dependency = dependency.strip()
            if dependency:
                subprocess.run(
                    ["pip", "install", "--target", package_dir, dependency], check=True
                )

        # Create a .zip file for the deployment package
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            # Add the installed dependencies to the .zip file
            for root, _, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate relative path for the zip file
                    arc_name = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arc_name)

            # Add the lambda_function.py file to the .zip file
            zipf.write(lambda_file_path, "lambda_function.py")

        return output_zip_path

    finally:
        # Clean up the project directory
        if os.path.exists(project_path):
            shutil.rmtree(project_path)


def get_tool_list():
    """Define and return the tool list for the LLM to use."""
    return [
        {
            "toolSpec": {
                "name": "create_lambda_function",
                "description": "Create and deploy a Lambda function.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code for the Lambda function.",
                            },
                            "function_name": {
                                "type": "string",
                                "description": "The name of the Lambda function.",
                            },
                            "description": {
                                "type": "string",
                                "description": "A description of the Lambda function.",
                            },
                            "has_external_python_libraries": {
                                "type": "boolean",
                                "description": "Whether the function uses external Python libraries.",
                            },
                            "external_python_libraries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of external Python libraries to include.",
                            },
                        },
                        "required": [
                            "code",
                            "function_name",
                            "description",
                            "has_external_python_libraries",
                            "external_python_libraries",
                        ],
                    }
                },
            }
        },
    ]


def query_llm(bedrock, messages, tools, system_prompt):
    """Make a request to the LLM and return the response."""
    return bedrock.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        inferenceConfig={"maxTokens": 2000, "temperature": 0.7},
        toolConfig={"tools": tools},
        system=[{"text": system_prompt}],
    )


def create_lambda_function(
    lambda_client,
    s3,
    code: str,
    function_name: str,
    description: str,
    has_external_python_libraries: bool,
    external_python_libraries: List[str],
) -> str:
    """
    Creates and deploys a Lambda Function, based on what the customer requested.
    Returns the name of the created Lambda function
    """
    print("Creating Lambda function")
    runtime = "python3.13"
    handler = "lambda_function.lambda_handler"

    # Create a zip file for the code
    if has_external_python_libraries:
        zipfile = create_deployment_package_with_dependencies(
            code, function_name, f"{function_name}.zip", external_python_libraries
        )
    else:
        zipfile = create_deployment_package_no_dependencies(
            code, function_name, f"{function_name}.zip"
        )

    try:
        # Upload zip file
        zip_key = f"lambda_resources/{function_name}.zip"
        s3.upload_file(zipfile, S3_BUCKET, zip_key)
        print(f"Uploaded zip to {S3_BUCKET}/{zip_key}")

        response = lambda_client.create_function(
            Code={
                "S3Bucket": S3_BUCKET,
                "S3Key": zip_key,
            },
            Description=description,
            FunctionName=function_name,
            Handler=handler,
            Timeout=30,
            Publish=True,
            Role=LAMBDA_ROLE,
            Runtime=runtime,
        )
        print("Lambda function created successfully")
        print(response)
        deployed_function = response["FunctionName"]
        user_response = f"The function {deployed_function} has been deployed to the customer's AWS account. I will now provide my final answer to the customer on how to invoke the {deployed_function} function with boto3 and print the result."
        return user_response
    except ClientError as e:
        print(e)
        return f"Error: {e}\n Let me try again..."


def process_llm_response(response_message, lambda_client, s3):
    """Process the LLM's response, handling tool usage and text output."""
    response_content_blocks = response_message["content"]
    follow_up_content_blocks = []
    # llm_response = ""

    for content_block in response_content_blocks:
        if "toolUse" in content_block:
            tool_use_block = content_block["toolUse"]
            tool_use_name = tool_use_block["name"]
            print(f"Using tool {tool_use_name}")
            if tool_use_name == "create_lambda_function":
                result = create_lambda_function(
                    lambda_client,
                    s3,
                    tool_use_block["input"]["code"],
                    tool_use_block["input"]["function_name"],
                    tool_use_block["input"]["description"],
                    tool_use_block["input"]["has_external_python_libraries"],
                    tool_use_block["input"]["external_python_libraries"],
                )
                print(f"Lambda function creation result: {result}")
                # llm_response = result
                follow_up_content_blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_block["toolUseId"],
                            "content": [{"json": {"result": result}}],
                        }
                    }
                )
        elif "text" in content_block:
            print(f"LLM response: {content_block['text']}")

            # llm_response = content_block["text"]

    return follow_up_content_blocks


def lambda_function_pipeline(input_text):
    # Initialize the AWS clients
    bedrock, lambda_client, s3 = initialize_clients()

    # Get the tool list
    tool_list = get_tool_list()

    # Initialize the message list for the conversation
    message_list = [
        {
            "role": "user",
            "content": [{"text": input_text}],
        }
    ]

    # Set the system prompt
    system_prompt = "You are an AI assistant capable of creating Python Lambda functions. Use the provided tools when necessary."

    # Make the initial request to the LLM
    response = query_llm(bedrock, message_list, tool_list, system_prompt)
    response_message = response["output"]["message"]
    print(json.dumps(response_message, indent=4))
    message_list.append(response_message)

    # Process the LLM's response
    follow_up_content_blocks = process_llm_response(response_message, lambda_client, s3)
    print("THIS IS FOLLOW UP CONTEXT")
    print(follow_up_content_blocks)

    # If there are follow-up content blocks, make another request to the LLM
    if follow_up_content_blocks:
        print("MAKING FOLLOW UP REQUEST")
        follow_up_message = {
            "role": "user",
            "content": follow_up_content_blocks,
        }
        message_list.append(follow_up_message)

        response = query_llm(bedrock, message_list, tool_list, system_prompt)
        response_message = response["output"]["message"]
        print(json.dumps(response_message, indent=4))
        message_list.append(response_message)

        # Process the final response
        follow_up_content_blocks = process_llm_response(
            response_message, lambda_client, s3
        )
    print("THIS IS MESSAGE LIST")
    print(message_list)
    return message_list


def lambda_handler(event, context):
    # Print the received event to the logs
    print("Received event: ")
    print(event)

    # Initialize response code to None

    # Extract the action group, api path, and parameters from the prediction
    actionGroup = event["actionGroup"]
    function = event.get("function", "")
    parameters = event.get("parameters", [])
    inputText = event.get("inputText", "")

    results = lambda_function_pipeline(inputText)

    response_body = {"TEXT": {"body": str(results)}}

    # Print the response body to the logs
    print(f"Response body: {response_body}")

    # Create a dictionary containing the response details
    action_response = {
        "actionGroup": actionGroup,
        "function": function,
        "functionResponse": {"responseBody": response_body},
    }

    # Return the list of responses as a dictionary
    api_response = {
        "messageVersion": event["messageVersion"],
        "response": action_response,
    }

    return api_response
