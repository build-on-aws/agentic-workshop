import json
import os

import boto3
import requests

# Can get an API KEY here: https://jina.ai/reader/
JINA_KEY = os.getenv("JINA_KEY")

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


def process_website(input_text, website_text):

    prompt = f"{input_text} <website text>{website_text}</website_text>"

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
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

    # Extract the action group, api path, and parameters from the prediction
    actionGroup = event["actionGroup"]
    function = event.get("function", "")
    parameters = event.get("parameters", [])
    inputText = event.get("inputText", "")
    website_url = parameters[0]["value"]

    url = f"https://r.jina.ai/{website_url}"
    headers = {"Authorization": f"Bearer {JINA_KEY}"}
    response = requests.get(url, headers=headers)

    # process request
    result = process_website(inputText, response.text)

    response_body = {"TEXT": {"body": result}}

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
