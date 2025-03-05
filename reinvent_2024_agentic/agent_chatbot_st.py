import os
import re
import uuid
from datetime import datetime
from io import BytesIO

import agent_tools
import boto3
import streamlit as st
from PIL import Image

# Initialize S3 client
s3_client = boto3.client("s3")

# Sample questions
SAMPLE_QUESTIONS = [
    "What are the best practices for cloud security?",
    "Can you draw an AWS diagram that shows an ecommerce architecture",
    "What are the top 5 stories from https://aws.amazon.com/blogs/aws/",
    "Can you create a lambda function that can do sentiment analysis on text?",
]


def upload_to_s3(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client("s3")
        bucket_name = os.getenv("S3_BUCKET_NAME")

        # Generate a unique file name to avoid collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"

        content_type = (
            "image/jpeg"
            if file_name.lower().endswith((".jpg", ".jpeg"))
            else "image/png"
        )

        s3_client.put_object(
            Bucket=bucket_name, Key=s3_key, Body=file_bytes, ContentType=content_type
        )

        url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        return url
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return None


def extract_and_display_s3_images(text, s3_client):
    """
    Extract S3 URLs from text, download images, and return them for display
    """
    s3_pattern = r"https://[\w\-\.]+\.s3\.amazonaws\.com/[\w\-\./]+"
    s3_urls = re.findall(s3_pattern, text)

    images = []
    for url in s3_urls:
        try:
            bucket = url.split(".s3.amazonaws.com/")[0].split("//")[1]
            key = url.split(".s3.amazonaws.com/")[1]

            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()

            image = Image.open(BytesIO(image_data))
            images.append(image)

        except Exception as e:
            st.error(f"Error downloading image from S3: {str(e)}")
            continue

    return images


def process_query(prompt, uploaded_file=None):
    """Handle the query processing and response"""
    # Check if there's an uploaded file
    image_url = None
    if uploaded_file is not None:
        # Upload the file to S3
        image_url = upload_to_s3(uploaded_file.getvalue(), uploaded_file.name)

        # If image was uploaded successfully, append it to the message
        if image_url:
            prompt = f"{prompt}\nhere is the image: {image_url}"

    # Add user message to chat
    st.session_state.messages.append(
        {
            "role": "user",
            "content": [{"text": prompt}],
            "images": [image_url] if image_url else [],
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        trace_container = st.container()

        result = agent_tools.invoke_bedrock_agent(
            prompt, st.session_state.session_id, trace_container
        )

        st.markdown(result["text"])

        if "images" in result:
            for image in result["images"]:
                if isinstance(image, str) and image.startswith("http"):
                    st.image(image)
                elif isinstance(image, Image.Image):
                    st.image(image, use_column_width=True)
                else:
                    image_data = Image.open(image)
                    st.image(image_data, use_column_width=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": [{"text": f"{result['text']}"}],
                "images": result["images"] if "images" in result else [],
                "traces": result["traces"] if "traces" in result else [],
            }
        )


st.title("Amazon Bedrock Agentic Chatbot")
st.sidebar.markdown(
    "This app shows an Agentic Chatbot powered by Amazon Bedrock to answer questions."
)

# Add file uploader to sidebar
st.sidebar.subheader("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

# Preview the uploaded image in the sidebar
if uploaded_file is not None:
    st.image(uploaded_file, caption="Preview of uploaded image", use_column_width=True)
    if st.button("Clear Image"):
        uploaded_file = None
        st.rerun()

clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Initialize session state for sample questions visibility
if "show_sample_questions" not in st.session_state:
    st.session_state.show_sample_questions = True

# Reset sessions state on clear
if clear_button:
    st.session_state.messages = []
    st.session_state.session_id = agent_tools.generate_random_15digit()
    st.session_state.show_sample_questions = (
        True  # Show sample questions again after clearing
    )

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.session_id = agent_tools.generate_random_15digit()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "traces" in message:
            trace_container = st.container()
            for trace in message["traces"]:
                with trace_container.expander(trace["trace_type"]):
                    if trace["trace_type"] == "codeInterpreter":
                        st.code(trace["text"], language="python")
                    else:
                        st.markdown(trace["text"])

        message_text = message["content"][0]["text"]
        st.markdown(message_text)

        # Display images in the message
        if "images" in message and message["images"]:
            for image_url in message["images"]:
                if image_url:  # Only display if image_url is not None
                    st.image(image_url)


# Display sample questions in a 2x2 grid if they should be shown
if st.session_state.show_sample_questions:
    st.write("Try asking one of these questions:")
    col1, col2 = st.columns(2)

    # First row
    if col1.button(SAMPLE_QUESTIONS[0], key="q1", use_container_width=True):
        st.session_state.show_sample_questions = False
        process_query(SAMPLE_QUESTIONS[0], uploaded_file)
        st.rerun()
    if col2.button(SAMPLE_QUESTIONS[1], key="q2", use_container_width=True):
        st.session_state.show_sample_questions = False
        process_query(SAMPLE_QUESTIONS[1], uploaded_file)
        st.rerun()

    # Second row
    if col1.button(SAMPLE_QUESTIONS[2], key="q3", use_container_width=True):
        st.session_state.show_sample_questions = False
        process_query(SAMPLE_QUESTIONS[2], uploaded_file)
        st.rerun()
    if col2.button(SAMPLE_QUESTIONS[3], key="q4", use_container_width=True):
        st.session_state.show_sample_questions = False
        process_query(SAMPLE_QUESTIONS[3], uploaded_file)
        st.rerun()

# Always show the chat input
if user_input := st.chat_input("How can I help??"):
    process_query(user_input, uploaded_file)
