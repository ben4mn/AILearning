# What Are AI Agents?

## Introduction

When you ask ChatGPT a question, it responds—and that's it. The interaction is one-shot: you provide input, you get output, and the system is done. But what if AI could do more? What if it could break down complex tasks, make plans, take actions in the world, observe results, and adjust its approach—more like a human assistant than a text generator?

This vision has given rise to **AI agents**: systems that combine the language understanding and reasoning of LLMs with the ability to take autonomous actions over extended interactions. In this lesson, we'll explore what makes something an "agent," the spectrum of autonomy levels, and how planning and action work together in modern AI systems.

## From Chat to Agency

### The Chat Paradigm

Traditional LLM interactions follow a simple pattern:

```
User: [Input]
LLM: [Output]
(Done)
```

The model processes your input once and generates a response. Even in a conversation, each exchange is essentially independent—the model doesn't take actions between turns or work toward longer-term goals.

### The Agent Paradigm

Agents follow a different pattern:

```
User: "Book me a flight to Tokyo next week, find a hotel near Shibuya,
      and add both to my calendar"

Agent:
├── [Thinks: I need to break this into steps]
├── [Action: Search flights to Tokyo]
├── [Observes: Several options found, ANA has best times]
├── [Action: Book ANA flight - March 15-22]
├── [Observes: Booking confirmed, ref #ABC123]
├── [Action: Search hotels near Shibuya]
├── [Observes: Found 5 options in budget range]
├── [Action: Book Hotel Shibuya - 7 nights]
├── [Observes: Reservation confirmed]
├── [Action: Create calendar events]
├── [Observes: Events added to calendar]
└── [Response: "I've booked your trip! Flight ANA 123 departing March 15..."]
```

The agent:
- **Plans**: Breaks down the complex request into subtasks
- **Acts**: Executes actions in external systems
- **Observes**: Processes results of actions
- **Adapts**: Adjusts based on what it learns
- **Persists**: Works across multiple steps toward a goal

## The Autonomy Spectrum

Not all agents are equally autonomous. There's a spectrum:

### Level 0: Pure LLM (No Agency)

```python
# Traditional chat - no agency
response = llm.generate(user_message)
return response
```

The model just generates text. No tools, no actions, no iteration.

### Level 1: Tool-Augmented Chat

```python
# Can use tools, but one-shot
if needs_calculation(user_message):
    result = calculator.compute(expression)
    response = llm.generate(f"The result is {result}")
elif needs_search(user_message):
    info = search_engine.query(query)
    response = llm.generate(f"Based on my search: {info}")
```

The model can use tools, but decisions are simple and execution is single-step.

### Level 2: Simple Agent Loop

```python
# Can reason and iterate
while not task_complete:
    thought = llm.generate("What should I do next?")
    action = parse_action(thought)
    result = execute_action(action)
    history.append((action, result))

    if is_final_answer(thought):
        task_complete = True
```

The agent can take multiple steps, observe results, and continue until done.

### Level 3: Autonomous Agent

```python
# Sets own goals, works independently
while True:
    # Agent determines what to work on
    goal = agent.prioritize_goals()

    # Plans approach
    plan = agent.create_plan(goal)

    # Executes with minimal oversight
    for step in plan:
        result = agent.execute(step)
        if needs_replanning(result):
            plan = agent.revise_plan(result)

    # Moves to next goal
    agent.complete_goal(goal)
```

The agent operates with minimal human oversight, setting and pursuing goals independently.

### Level 4: Multi-Agent Systems

```python
# Multiple agents collaborate
agents = [ResearchAgent(), WriterAgent(), EditorAgent()]

research = agents[0].research(topic)
draft = agents[1].write(research)
final = agents[2].edit(draft)
```

Multiple specialized agents work together on complex tasks.

## The Agent Loop

Most agent architectures follow a common pattern:

```
┌──────────────────────────────────────────────────────────────┐
│                       AGENT LOOP                              │
│                                                               │
│    ┌─────────────┐                                           │
│    │   THINK     │ ← What do I know? What should I do?       │
│    └──────┬──────┘                                           │
│           │                                                   │
│           ▼                                                   │
│    ┌─────────────┐                                           │
│    │   PLAN      │ ← Break task into steps                   │
│    └──────┬──────┘                                           │
│           │                                                   │
│           ▼                                                   │
│    ┌─────────────┐                                           │
│    │   ACT       │ ← Execute an action (use a tool)          │
│    └──────┬──────┘                                           │
│           │                                                   │
│           ▼                                                   │
│    ┌─────────────┐                                           │
│    │   OBSERVE   │ ← What was the result?                    │
│    └──────┬──────┘                                           │
│           │                                                   │
│           ▼                                                   │
│    ┌─────────────┐                                           │
│    │   REFLECT   │ ← Am I done? What did I learn?            │
│    └──────┬──────┘                                           │
│           │                                                   │
│           │ not done                                          │
│           └────────────────────────────┐                      │
│                                        │                      │
│           ┌────────────────────────────┘                      │
│           │                                                   │
│           ▼                                                   │
│        (back to THINK)                                        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class Agent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []

    def run(self, task, max_steps=10):
        self.memory.append({"role": "user", "content": task})

        for step in range(max_steps):
            # THINK: Generate reasoning and action
            response = self.llm.generate(
                self.build_prompt(),
                stop=["Observation:"]
            )
            self.memory.append({"role": "assistant", "content": response})

            # Parse what the agent wants to do
            if "Final Answer:" in response:
                return self.extract_answer(response)

            if "Action:" in response:
                # ACT: Execute the action
                action, input = self.parse_action(response)
                tool = self.tools[action]
                observation = tool.execute(input)

                # OBSERVE: Record result
                self.memory.append({
                    "role": "system",
                    "content": f"Observation: {observation}"
                })

        return "Max steps reached without answer"
```

## Why LLMs Enable Agents

Before LLMs, building agents was extremely difficult:

**Planning**: Required explicit symbolic planners with carefully defined operators
**Language understanding**: Couldn't understand natural language goals
**Flexibility**: Brittle—broke on unexpected situations
**Tool use**: Required manual integration for each tool

LLMs change everything:

**Planning**: Can decompose tasks expressed in natural language
**Language understanding**: Native capability
**Flexibility**: Handle unexpected situations through reasoning
**Tool use**: Can learn tool usage from descriptions

```python
# LLM can understand and use tools from descriptions alone
tool_description = """
Available tools:
- search(query): Search the web and return results
- calculate(expression): Evaluate a mathematical expression
- email(to, subject, body): Send an email

To use a tool, write: Action: tool_name(arguments)
"""

# LLM can figure out when and how to use these
# without explicit programming for each case
```

## What Agents Can and Cannot Do

### Current Capabilities

Agents can effectively:
- **Research**: Gather information from multiple sources
- **Code**: Write, test, debug, and iterate on programs
- **Data analysis**: Query databases, process data, generate reports
- **Automation**: Perform multi-step workflows across APIs
- **Assistance**: Handle complex requests that require multiple tools

### Current Limitations

Agents struggle with:
- **Long-term planning**: Maintaining coherent plans over many steps
- **Error recovery**: Gracefully handling unexpected failures
- **Consistency**: Staying on track without getting confused
- **Reliability**: Guaranteeing correct completion of critical tasks
- **Cost**: Extended reasoning is expensive in tokens and time

### The Reliability Challenge

```python
# Agents can fail in various ways
challenges = [
    "Getting stuck in loops",
    "Forgetting the original goal",
    "Making errors that compound",
    "Hallucinating tool capabilities",
    "Getting confused by unexpected results",
    "Running up large token costs",
]

# Current agents need guardrails
# - Step limits
# - Human oversight for critical actions
# - Sandboxed execution environments
# - Logging and monitoring
```

## Key Takeaways

1. **AI agents extend LLMs with action-taking capability**, enabling multi-step task completion rather than single-turn responses.

2. **Agency exists on a spectrum** from simple tool use to fully autonomous goal-directed behavior.

3. **The agent loop—think, plan, act, observe, reflect**—is the fundamental pattern underlying most agent architectures.

4. **LLMs enable flexible agency** through natural language understanding, reasoning, and ability to learn tool use from descriptions.

5. **Current agents have real capabilities but also real limitations**, particularly around reliability and long-term coherence.

6. **Human oversight remains important** for critical applications where agent errors could have significant consequences.

## Further Reading

- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "AutoGPT" and "BabyAGI" project documentation
- "Language Models as Zero-Shot Planners" (Huang et al., 2022)
- "Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)
- "The Rise and Potential of Large Language Model Based Agents" (Wang et al., 2023)
