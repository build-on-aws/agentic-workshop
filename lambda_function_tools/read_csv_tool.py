import json
import os

import boto3
import pandas

S3_BUCKET = os.environ["S3_BUCKET"]
S3_OBJECT = os.environ["S3_OBJECT"]


def lambda_handler(event, context):
    # Print the received event to the logs
    print("Received event: ")
    print(event)

    # Initialize response code to None
    response_code = None

    # Extract the action group, api path, and parameters from the prediction
    agent = event["agent"]
    actionGroup = event["actionGroup"]
    function = event["function"]
    parameters = event.get("parameters", [])

    # Check the api path to determine which tool function to call

    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_OBJECT, "/tmp/data.csv")
    df = pandas.read_csv("/tmp/data.csv")

    # Get count of dataframe
    count = len(df)

    response_body = {"TEXT": {"body": str(count)}}
    response_code = 200

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
