import base64
import io
import json
from typing import Any, Dict, List, Type, Union

import boto3
from PIL import Image

s3 = boto3.client("s3")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


# function to convert a PIL image to a base64 string
def pil_to_base64(image, format="png"):
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        return base64.b64encode(buffer.getvalue()).decode()


def gen_image_caption(base64_string):

    system_prompt = """

    You are an experienced AWS Solutions Architect with deep knowledge of AWS services and best practices for designing and implementing cloud architectures. Maintain a professional and consultative tone, providing clear and detailed explanations tailored for technical audiences. Your task is to describe and explain AWS architecture diagrams presented by users. Your descriptions should cover the purpose and functionality of the included AWS services, their interactions, data flows, and any relevant design patterns or best practices.
    """

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_string,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Please describe the following AWS architecture diagram, explaining the purpose of each service, their interactions, and any relevant design considerations or best practices.",
                    },
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    modelId = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results


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

    image_url = parameters[0]["value"]

    # Download image from s3

    bucket_name = image_url.split("/")[2].split(".")[0]
    key = "/".join(image_url.split("/")[3:])
    response = s3.get_object(Bucket=bucket_name, Key=key)
    image_content = response["Body"].read()
    # Create a PIL Image object from the image content
    image = Image.open(io.BytesIO(image_content))
    # Convert the PIL Image object to a base64 string
    base64_string = pil_to_base64(image)

    results = gen_image_caption(base64_string)

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
