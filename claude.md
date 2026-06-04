# Claude / Prompting Cheat Sheet

## Prompt Engineering Techniques

### Chain of Thought
Ask Claude to reason step by step before answering.
"Think through this step by step before giving your final answer."

### Few-Shot Prompting
Give examples before your actual request.
"Here are two examples of what I want:
Example 1: ...
Example 2: ...
Now do the same for: ..."

### Role Prompting
"You are a senior software engineer at a FAANG company.
Review this code and give brutal honest feedback."

### Output Formatting with XML
"Respond using this structure:
<analysis>your analysis here</analysis>
<recommendation>your recommendation here</recommendation>
<confidence>high/medium/low</confidence>"

### Negative Prompting
Tell Claude what NOT to do.
"Explain quantum computing. Do not use jargon.
Do not use analogies involving cats."

### Iterative Refinement
"That's good. Now make it more concise."
"Rewrite the second paragraph to be more persuasive."
"Give me 3 alternative versions of that."

---

## Claude API — Advanced Usage (Python)

### Basic Setup
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

### System Prompts
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a no-nonsense financial analyst. Be concise and data-driven.",
    messages=[
        {"role": "user", "content": "Should I invest in ETFs?"}
    ]
)

### Multi-Turn Conversations
conversation = []

def chat(user_input):
    conversation.append({"role": "user", "content": user_input})
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=conversation
    )
    reply = response.content[0].text
    conversation.append({"role": "assistant", "content": reply})
    return reply

### Streaming Responses
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a long essay on AI"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

### Tool Use (Function Calling)
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Brisbane?"}]
)

### Vision — Sending Images
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }]
)

### Batch Processing
import anthropic

batch = client.messages.batches.create(
    requests=[
        {"custom_id": "q1", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [{"role": "user", "content": "What is AI?"}]}},
        {"custom_id": "q2", "params": {"model": "claude-sonnet-4-20250514", "max_tokens": 100, "messages": [{"role": "user", "content": "What is ML?"}]}},
    ]
)

---

## Models Comparison
| Model              | Speed    | Intelligence | Best For                  |
|--------------------|----------|--------------|---------------------------|
| claude-opus-4-6    | Slow     | Highest      | Complex reasoning, research|
| claude-sonnet-4-6  | Balanced | High         | Most tasks (recommended)  |
| claude-haiku-4-5   | Fastest  | Good         | Simple tasks, high volume |

---

## Token & Cost Tips
- max_tokens controls response length, not input length
- Shorter system prompts = lower cost
- Use Haiku for classification/routing, Sonnet for generation
- Cache repeated system prompts with prompt caching to save ~90% cost

## Prompt Caching (Cost Saving)
system=[{
    "type": "text",
    "text": "your long system prompt here...",
    "cache_control": {"type": "ephemeral"}
}]

---

## Safety & Best Practices
- Never hardcode API keys — always use .env
- Validate and sanitize user input before sending to API
- Set appropriate max_tokens to control costs
- Use temperature=0 for deterministic tasks (classification, extraction)
- Log inputs/outputs in production for debugging