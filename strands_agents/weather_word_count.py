from strands import Agent, tool
from strands.models import BedrockModel
from strands_tools import http_request

# Define a weather-focused system prompt
WEATHER_SYSTEM_PROMPT = """You are a weather assistant with HTTP capabilities. You can:

1. Make HTTP requests to the National Weather Service API
2. Process and display weather forecast data
3. Provide weather information for locations in the United States

When retrieving weather information:
1. First get the coordinates or grid information using https://api.weather.gov/points/{latitude},{longitude} or https://api.weather.gov/points/{zipcode}
2. Then use the returned forecast URL to get the actual forecast

When displaying responses:
- Format weather data in a human-readable way
- Highlight important information like temperature, precipitation, and alerts
- Handle errors appropriately
- Convert technical terms to user-friendly language

Always explain the weather conditions clearly and provide context for the forecast.
"""


@tool
def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


# Bedrock
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
    temperature=0.3,
)

agent = Agent(
    system_prompt=WEATHER_SYSTEM_PROMPT,
    tools=[word_count, http_request],
    model=bedrock_model,
)
response = agent(
    "What's the weather like in Seattle? Also how many words are in the response?"
)
