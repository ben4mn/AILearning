# Tool Use: Function Calling, APIs, and Code Execution

## Introduction

An LLM in isolation is powerful but limited—it can only generate text. It can't check the weather, query a database, send an email, or browse the web. **Tool use** bridges this gap, giving language models the ability to interact with the world through well-defined interfaces.

In this lesson, we'll explore how modern LLMs use tools: the function calling paradigm, integration with APIs, code execution environments, and best practices for building tool-enabled systems.

## The Tool-Use Paradigm

### From Text to Action

The core idea is simple: instead of just generating text, the LLM can decide to call a function:

```
User: "What's the weather in Tokyo?"

Traditional LLM:
"I don't have access to real-time weather data, but typically
in March, Tokyo experiences..."

Tool-enabled LLM:
[Decides to use weather tool]
Action: get_weather("Tokyo")
Observation: {"temp": 15, "conditions": "partly cloudy"}
Response: "It's currently 15°C and partly cloudy in Tokyo."
```

The model reasons about *when* to use a tool and *what* inputs to provide, then incorporates the results into its response.

## Function Calling

### How It Works

Modern LLM APIs support structured function calling:

```python
import openai

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Tokyo' or 'London'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Call the API
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"  # Let model decide when to use tools
)

# Check if model wants to use a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
    # Output: Function: get_weather
    #         Arguments: {"location": "Tokyo", "unit": "celsius"}
```

### Processing Tool Results

After getting tool results, feed them back to the model:

```python
def complete_with_tools(user_message, tools, available_functions):
    messages = [{"role": "user", "content": user_message}]

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )

    assistant_message = response.choices[0].message

    # If model wants to use tools
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # Execute each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Actually call the function
            function = available_functions[function_name]
            result = function(**arguments)

            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Get final response incorporating tool results
        final_response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )

        return final_response.choices[0].message.content

    return assistant_message.content
```

## Common Tool Categories

### Information Retrieval

```python
tools = [
    {
        "name": "search_web",
        "description": "Search the internet for current information",
        "parameters": {"query": "string"}
    },
    {
        "name": "search_documents",
        "description": "Search internal knowledge base",
        "parameters": {"query": "string", "category": "string?"}
    },
    {
        "name": "get_url_content",
        "description": "Fetch and read content from a URL",
        "parameters": {"url": "string"}
    }
]
```

### Data Operations

```python
tools = [
    {
        "name": "query_database",
        "description": "Run SQL query on the database",
        "parameters": {"sql": "string"}
    },
    {
        "name": "analyze_data",
        "description": "Perform statistical analysis",
        "parameters": {"data": "array", "analysis_type": "string"}
    },
    {
        "name": "create_chart",
        "description": "Generate a visualization",
        "parameters": {"data": "object", "chart_type": "string"}
    }
]
```

### Communication

```python
tools = [
    {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {"to": "string", "subject": "string", "body": "string"}
    },
    {
        "name": "send_slack_message",
        "description": "Post message to Slack channel",
        "parameters": {"channel": "string", "message": "string"}
    },
    {
        "name": "schedule_meeting",
        "description": "Add meeting to calendar",
        "parameters": {"title": "string", "time": "datetime", "attendees": "array"}
    }
]
```

### Computation

```python
tools = [
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "string"}
    },
    {
        "name": "execute_code",
        "description": "Run Python code and return results",
        "parameters": {"code": "string"}
    }
]
```

## Code Execution

### Why Code Execution?

Instead of defining every possible operation as a tool, let the model write code:

```python
# Limited tools approach
def calculate(expression): ...
def search_data(query): ...
def filter_data(data, condition): ...
# Need a tool for everything!

# Code execution approach
def execute_python(code):
    """Run arbitrary Python code in sandbox."""
    return sandbox.run(code)

# Model can now do anything Python can do:
# - Complex calculations
# - Data manipulation
# - Custom logic
# - Combining multiple operations
```

### Safe Code Execution

Running arbitrary code requires careful sandboxing:

```python
import subprocess
import resource
import os

class SafeCodeExecutor:
    def __init__(self, timeout=30, max_memory_mb=512):
        self.timeout = timeout
        self.max_memory = max_memory_mb * 1024 * 1024

    def execute(self, code):
        """Execute Python code in isolated environment."""

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name

        try:
            # Run in subprocess with limits
            result = subprocess.run(
                ['python', code_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                # Run in restricted environment
                env={
                    'PATH': '/usr/bin',
                    'HOME': '/tmp',
                    # No network access, limited filesystem
                }
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {"error": "Execution timed out"}
        finally:
            os.unlink(code_file)
```

### Container-Based Execution

For production systems, use containers:

```python
import docker

class DockerCodeExecutor:
    def __init__(self):
        self.client = docker.from_env()

    def execute(self, code):
        """Execute code in isolated Docker container."""
        container = self.client.containers.run(
            "python:3.11-slim",
            command=["python", "-c", code],
            detach=True,
            mem_limit="512m",
            cpu_period=100000,
            cpu_quota=50000,  # 50% CPU
            network_mode="none",  # No network
            read_only=True,  # Read-only filesystem
        )

        try:
            container.wait(timeout=30)
            logs = container.logs()
            return logs.decode('utf-8')
        finally:
            container.remove(force=True)
```

## Building Robust Tool Systems

### Tool Description Quality

Good descriptions help the model use tools correctly:

```python
# Bad: Vague description
{
    "name": "search",
    "description": "Search for stuff",
    "parameters": {"q": "string"}
}

# Good: Clear and specific
{
    "name": "search_product_catalog",
    "description": "Search the product catalog by name, category, or SKU. Returns up to 10 matching products with name, price, and availability.",
    "parameters": {
        "query": {
            "type": "string",
            "description": "Search terms (product name, category, or SKU)"
        },
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "home", "books"],
            "description": "Optional: Filter by product category"
        },
        "in_stock_only": {
            "type": "boolean",
            "description": "If true, only return products currently in stock"
        }
    }
}
```

### Error Handling

Tools fail. Handle it gracefully:

```python
def execute_tool_safely(tool_name, arguments, available_tools):
    """Execute tool with comprehensive error handling."""
    try:
        tool = available_tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}

        result = tool(**arguments)
        return {"success": True, "result": result}

    except TypeError as e:
        return {
            "error": "Invalid arguments",
            "details": str(e),
            "expected": get_tool_signature(tool_name)
        }

    except PermissionError:
        return {"error": "Permission denied for this operation"}

    except TimeoutError:
        return {"error": "Operation timed out"}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# The LLM can understand and adapt to errors
# "The search failed due to timeout. Let me try a more specific query..."
```

### Rate Limiting and Costs

Tools can be expensive or rate-limited:

```python
from functools import wraps
import time

class ToolRateLimiter:
    def __init__(self):
        self.call_times = {}

    def rate_limited(self, calls_per_minute):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func.__name__
                now = time.time()

                # Clean old entries
                self.call_times[name] = [
                    t for t in self.call_times.get(name, [])
                    if now - t < 60
                ]

                # Check limit
                if len(self.call_times[name]) >= calls_per_minute:
                    raise RateLimitError("Rate limit exceeded")

                # Record this call
                self.call_times[name].append(now)

                return func(*args, **kwargs)
            return wrapper
        return decorator

limiter = ToolRateLimiter()

@limiter.rate_limited(calls_per_minute=10)
def expensive_api_call(query):
    return external_api.search(query)
```

## Multi-Tool Orchestration

Complex tasks require multiple tools:

```python
# User: "Find recent papers about transformer efficiency,
#        summarize the top 3, and email me the summary"

# Agent orchestrates multiple tools:
steps = [
    ("search_arxiv", {"query": "transformer efficiency", "year": 2024}),
    ("fetch_paper", {"url": results[0]["url"]}),
    ("fetch_paper", {"url": results[1]["url"]}),
    ("fetch_paper", {"url": results[2]["url"]}),
    ("summarize_documents", {"documents": papers}),
    ("send_email", {"to": user_email, "subject": "Paper Summary", "body": summary})
]

# Each step uses the results of previous steps
```

### Parallel Tool Calls

Some tools can run in parallel:

```python
import asyncio

async def parallel_tool_execution(tool_calls):
    """Execute independent tool calls in parallel."""
    tasks = []
    for call in tool_calls:
        task = asyncio.create_task(
            execute_tool_async(call.name, call.arguments)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Model requests weather for multiple cities
# Execute all requests in parallel instead of sequentially
```

## Key Takeaways

1. **Function calling gives LLMs structured tool access**, allowing them to interact with external systems through well-defined interfaces.

2. **Good tool descriptions are essential**—the model uses them to decide when and how to use each tool.

3. **Code execution is a powerful meta-tool** that enables arbitrary computation, but requires careful sandboxing.

4. **Robust error handling helps models adapt** when tools fail or return unexpected results.

5. **Tool orchestration enables complex workflows** by combining multiple tools across steps.

6. **Security and rate limiting are essential** for production tool-enabled systems.

## Further Reading

- OpenAI Function Calling documentation
- Anthropic Tool Use documentation
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
- "Gorilla: Large Language Model Connected with Massive APIs" (Patil et al., 2023)
- LangChain and LlamaIndex documentation on tool integration
