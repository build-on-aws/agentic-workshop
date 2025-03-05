# Building an AWS Solutions Architect Agentic App with Amazon Bedrock

This project implements an agentic chatbot using Amazon Bedrock and AWS Lambda, capable of processing images, generating AWS architecture diagrams, and analyzing websites.

The chatbot leverages various AWS services to provide a rich, interactive experience. It can answer questions, display images, generate AWS architecture diagrams based on user requests, and analyze website content. The system is built using Streamlit for the frontend and integrates seamlessly with AWS services like S3, Lambda, and Bedrock.

This is the video that explains the code:
https://www.youtube.com/watch?v=XPHOybnXCd4

## Repository Structure

- `agent_tools.py`: Contains utility functions for the chatbot, including image processing and Bedrock agent interactions.
- `chatbot_st.py`: The main Streamlit application file for the chatbot interface.
- `lambda_functions/`: Directory containing Lambda function implementations:
  * `create_lambda_functions.py`: Creates and deploys Lambda functions dynamically.
  * `describe_image.py`: Generates captions for images stored in S3.
  * `gen_aws_diag_docker/`: Contains files for generating AWS architecture diagrams:
    - `diag_mapping.json`: Maps AWS service names to diagram categories.
    - `lambda_handler.py`: Handles the diagram generation process.
  * `website_to_text.py`: Extracts and processes text content from websites.
- `lambda_layers/`: Scripts for creating Lambda layers:
  * `make_pil_layer.sh`: Creates a layer for the Pillow library.
  * `make_requests_layer.sh`: Creates a layer for the Requests library.
- `test.py`: Contains test code for sentiment analysis using AWS Lambda.

## Usage Instructions

You will need to setup the agent in Amazon Bedrock

Follow the workshop instructions, to learn how to configure the agent and create action groups.

https://catalog.workshops.aws/building-agentic-workflows/en-US/chatbot-agent

Once your agent and action groups are setup you can replace `AGENT_ID` in agent_tools.py with your Agent ID.

### Running the Chatbot

To start the Streamlit application:

```
streamlit run agent_chatbot_st.py
```

This will launch the chatbot interface in your default web browser.

### Creating Lambda Layers

To create Lambda layers for Pillow and Requests libraries:

1. Navigate to the `lambda_layers` directory:
   ```
   cd lambda_layers
   ```

2. Run the shell scripts:
   ```
   ./make_pil_layer.sh
   ./make_requests_layer.sh
   ```

3. Upload the resulting ZIP files (`pillow-layer.zip` and `requests-layer.zip`) to AWS Lambda as layers.
