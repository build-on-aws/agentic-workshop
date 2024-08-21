import io
import random

import boto3
import matplotlib.pyplot as plt
import streamlit as st

AGENT_ID = "REPLACE_WITH_YOUR_AGENT_ID"
REGION = "us-west-2"
IMAGE_FOLDER = "images"

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=REGION,
)

bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime", region_name=REGION
)


def generate_random_15digit():
    number = ""

    for _ in range(15):
        number += str(random.randint(0, 9))

    return number


def invoke_bedrock_agent(inputText, sessionId, trace_container, endSession=False):
    # Invoke the Bedrock agent with the given input text
    response = bedrock_agent_runtime.invoke_agent(
        agentAliasId="TSTALIASID",
        agentId=AGENT_ID,
        sessionId=sessionId,
        inputText=inputText,
        endSession=endSession,
        enableTrace=True,
    )

    # Get the event stream from the response
    event_stream = response["completion"]

    model_response = {"text": "", "images": [], "files": [], "traces": []}

    # Process each event in the stream
    for index, event in enumerate(event_stream):
        print(f"Event {index}:")
        print(str(event))
        print("\n")

        try:
            # Check trace
            if "trace" in event:
                if (
                    "trace" in event["trace"]
                    and "orchestrationTrace" in event["trace"]["trace"]
                ):
                    trace_event = event["trace"]["trace"]["orchestrationTrace"]
                    if "rationale" in trace_event:
                        trace_text = trace_event["rationale"]["text"]
                        trace_object = {"trace_type": "rationale", "text": trace_text}
                        model_response["traces"].append(trace_object)

                        with trace_container.expander("rationale"):
                            st.markdown(trace_text)

                    # for invocationInput type
                    if "invocationInput" in trace_event:
                        if (
                            "codeInterpreterInvocationInput"
                            in trace_event["invocationInput"]
                        ):
                            trace_code = trace_event["invocationInput"][
                                "codeInterpreterInvocationInput"
                            ]["code"]
                            trace_object = {
                                "trace_type": "codeInterpreter",
                                "text": trace_code,
                            }
                            model_response["traces"].append(trace_object)

                            with trace_container.expander("codeInterpreter"):
                                st.code(trace_code)
                        if "knowledgeBaseLookupInput" in trace_event["invocationInput"]:
                            trace_text = trace_event["invocationInput"][
                                "knowledgeBaseLookupInput"
                            ]["text"]
                            trace_object = {
                                "trace_type": "knowledgeBaseLookup",
                                "text": trace_text,
                            }
                            model_response["traces"].append(trace_object)

                            with trace_container.expander("knowledgeBaseLookup"):
                                st.markdown(trace_text)

                        if (
                            "actionGroupInvocationInput"
                            in trace_event["invocationInput"]
                        ):
                            trace_text = trace_event["invocationInput"][
                                "actionGroupInvocationInput"
                            ]["function"]
                            trace_object = {
                                "trace_type": "actionGroupInvocation",
                                "text": trace_text,
                            }
                            model_response["traces"].append(trace_object)

                            with trace_container.expander("actionGroupInvocation"):
                                st.markdown(f"Calling function: {trace_text}")

                    # for observation type
                    if "observation" in trace_event:
                        if (
                            "codeInterpreterInvocationOutput"
                            in trace_event["observation"]
                        ):
                            if (
                                "executionOutput"
                                in trace_event["observation"][
                                    "codeInterpreterInvocationOutput"
                                ]
                            ):
                                trace_resp = trace_event["observation"][
                                    "codeInterpreterInvocationOutput"
                                ]["executionOutput"]
                                trace_object = {
                                    "trace_type": "observation",
                                    "text": trace_resp,
                                }
                                model_response["traces"].append(trace_object)

                                with trace_container.expander("observation"):
                                    st.markdown(trace_resp)
                            if (
                                "executionError"
                                in trace_event["observation"][
                                    "codeInterpreterInvocationOutput"
                                ]
                            ):
                                trace_resp = trace_event["observation"][
                                    "codeInterpreterInvocationOutput"
                                ]["executionError"]
                                trace_object = {
                                    "trace_type": "observation",
                                    "text": trace_resp,
                                }
                                model_response["traces"].append(trace_object)

                                with trace_container.expander("observation"):
                                    st.error(trace_resp)

                        if "knowledgeBaseLookupOutput" in trace_event["observation"]:
                            # trace_text = trace_event["observation"][
                            #     "knowledgeBaseLookupOutput"
                            # ]["text"]
                            trace_object = {
                                "trace_type": "knowledgeBaseLookupOutput",
                                "text": trace_event["observation"][
                                    "knowledgeBaseLookupOutput"
                                ]["retrievedReferences"],
                            }
                            model_response["traces"].append(trace_object)

                            with trace_container.expander("knowledgeBaseLookupOutput"):
                                # st.markdown(trace_text)

                                if (
                                    "retrievedReferences"
                                    in trace_event["observation"][
                                        "knowledgeBaseLookupOutput"
                                    ]
                                ):
                                    references = trace_event["observation"][
                                        "knowledgeBaseLookupOutput"
                                    ]["retrievedReferences"]
                                    for reference in references:
                                        st.markdown(
                                            f'{reference["location"]["s3Location"]["uri"]}'
                                        )
                                        st.markdown(f'{reference["content"]["text"]}')

                        if "actionGroupInvocationOutput" in trace_event["observation"]:
                            trace_resp = trace_event["observation"][
                                "actionGroupInvocationOutput"
                            ]["text"]
                            trace_object = {
                                "trace_type": "observation",
                                "text": trace_resp,
                            }
                            model_response["traces"].append(trace_object)

                            with trace_container.expander("observation"):
                                st.markdown(trace_resp)

                        if "finalResponse" in trace_event["observation"]:
                            trace_resp = trace_event["observation"]["finalResponse"][
                                "text"
                            ]
                            trace_object = {
                                "trace_type": "finalResponse",
                                "text": trace_resp,
                            }
                            model_response["traces"].append(trace_object)

                            with trace_container.expander("finalResponse"):
                                st.markdown(trace_resp)

                elif "guardrailTrace" in event["trace"]["trace"]:

                    guardrail_trace = event["trace"]["trace"]["guardrailTrace"]
                    if "inputAssessments" in guardrail_trace:
                        assessments = guardrail_trace["inputAssessments"]
                        for assessment in assessments:
                            if "contentPolicy" in assessment:
                                filters = assessment["contentPolicy"]["filters"]
                                for filter in filters:
                                    if filter["action"] == "BLOCKED":
                                        st.error(
                                            f"Guardrail blocked {filter['type']} confidence: {filter['confidence']}"
                                        )
                            if "topicPolicy" in assessment:
                                topics = assessment["topicPolicy"]["topics"]
                                for topic in topics:
                                    if topic["action"] == "BLOCKED":
                                        st.error(
                                            f"Guardrail blocked topic {topic['name']}"
                                        )
            # Handle text chunks
            if "chunk" in event:
                chunk = event["chunk"]
                if "bytes" in chunk:
                    text = chunk["bytes"].decode("utf-8")
                    print(f"Chunk: {text}")
                    model_response["text"] += text
                    return model_response

            # Handle file outputs
            if "files" in event:
                print("Files received")
                files = event["files"]["files"]
                for file in files:
                    name = file["name"]
                    type = file["type"]
                    bytes_data = file["bytes"]

                    # Display PNG images using matplotlib
                    if type == "image/png":

                        # save image to disk
                        img = plt.imread(io.BytesIO(bytes_data))
                        img_name = f"{IMAGE_FOLDER}/{name}"
                        plt.imsave(img_name, img)

                        # if image name not in images
                        if img_name not in model_response["images"]:
                            model_response["images"].append(img_name)
                        print(f"Image '{name}' saved to disk.")
                    # Save other file types to disk
                    else:
                        with open(name, "wb") as f:
                            f.write(bytes_data)
                            model_response["files"].append(name)
                        print(f"File '{name}' saved to disk.")
        except Exception as e:
            print(f"Error processing event: {e}")
            continue
