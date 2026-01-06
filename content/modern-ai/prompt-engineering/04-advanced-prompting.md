# Advanced Prompting Techniques

## Introduction

Beyond the foundational techniques of zero-shot, few-shot, and chain-of-thought prompting, researchers have developed increasingly sophisticated methods to push LLM capabilities further. These advanced techniques address specific limitations: complex multi-step reasoning, consistency, structured exploration of solution spaces, and reliable output generation.

In this lesson, we'll explore cutting-edge prompting strategies: ReAct for combining reasoning with action, Tree-of-Thoughts for exploring multiple paths, self-consistency for robust answers, and structured output techniques. These methods represent the current frontier of prompt engineering.

## ReAct: Reasoning and Acting

### The Core Idea

ReAct (Reasoning + Acting) interleaves chain-of-thought reasoning with concrete actions like searching, calculating, or retrieving information. The model explicitly states its thinking, takes an action, observes the result, and continues reasoning.

```
Question: What is the population of the capital of Australia?

Thought 1: I need to find the capital of Australia first.
Action 1: Search[capital of Australia]
Observation 1: The capital of Australia is Canberra.

Thought 2: Now I need to find the population of Canberra.
Action 2: Search[population of Canberra]
Observation 2: Canberra has a population of approximately 453,558 (2021).

Thought 3: I have the answer now.
Answer: The population of Canberra, the capital of Australia, is
approximately 453,558.
```

### Why ReAct Works

Traditional CoT reasoning happens entirely "in the model's head"—prone to hallucination when facts are needed. ReAct grounds reasoning in external information:

```python
# ReAct prompt template
react_prompt = """
Answer the following questions by reasoning step-by-step.
You can use these actions:
- Search[query]: Search for information
- Calculate[expression]: Perform calculations
- Lookup[term]: Look up a specific term from previous results

Format:
Thought: [your reasoning]
Action: [action with input]
Observation: [result of action]
... (repeat as needed)
Answer: [final answer]

Question: {question}
"""

# In a real system, actions are executed by code
def execute_action(action, input_text):
    if action == "Search":
        return search_engine.query(input_text)
    elif action == "Calculate":
        return eval(input_text)  # (with proper sandboxing!)
    elif action == "Lookup":
        return knowledge_base.lookup(input_text)
```

### ReAct in Practice

ReAct is particularly powerful for:
- Multi-hop questions requiring multiple facts
- Math problems needing precise calculation
- Tasks involving external tools or databases
- Research-style tasks combining multiple sources

```
Question: Is the Eiffel Tower taller than the Statue of Liberty,
and by how much?

Thought 1: I need heights of both structures.
Action 1: Search[Eiffel Tower height]
Observation 1: The Eiffel Tower is 330 meters (1,083 ft) tall.

Thought 2: Now I need the Statue of Liberty's height.
Action 2: Search[Statue of Liberty height]
Observation 2: The Statue of Liberty is 93 meters (305 ft) from
ground to torch tip.

Thought 3: Let me calculate the difference.
Action 3: Calculate[330 - 93]
Observation 3: 237

Thought 4: I can now answer the question.
Answer: Yes, the Eiffel Tower (330m) is taller than the Statue
of Liberty (93m) by 237 meters.
```

## Tree-of-Thoughts (ToT)

### Beyond Linear Reasoning

Chain-of-thought follows a single reasoning path. But some problems require exploring multiple possibilities, backtracking when paths fail, or comparing different approaches. **Tree-of-Thoughts** addresses this by explicitly exploring a tree of reasoning paths.

```
Problem: Using the numbers 4, 8, 6, form an expression that equals 24.

Branch 1: Try multiplication first
├── 4 × 8 = 32... too high, need to reduce by 8
├── 4 × 6 = 24... but 8 is unused ✗

Branch 2: Try combinations with 8
├── 8 × 6 = 48... need to get 24, divide by 2?
│   └── (8 × 6) / 2 = 24... but 4 ≠ 2 ✗
├── 8 × 4 = 32... need 24, subtract 8
│   └── (8 × 4) - 8 = 24... but 6 is unused ✗

Branch 3: Try using all three with operations
├── 4 × 6 = 24, times 8 ÷ 8 = 24 ✓
│   └── Solution: (4 × 6) × (8 ÷ 8) = 24 × 1 = 24 ✓
└── 8 + 8 + 8 = 24, can we make 8 from 4 and 6?
    └── 4 + 6 = 10 ≠ 8 ✗

Best solution: 4 × 6 × 8 ÷ 8 = 24
```

### Implementing ToT

```python
def tree_of_thoughts(problem, max_depth=5, branches=3):
    """Generate and evaluate multiple reasoning paths."""

    def generate_thoughts(state, depth):
        if depth == 0 or is_solution(state):
            return [(evaluate(state), state)]

        # Generate multiple next thoughts
        prompt = f"Given: {state}\nSuggest {branches} possible next steps:"
        thoughts = llm.generate(prompt)

        results = []
        for thought in thoughts:
            new_state = state + " → " + thought
            # Recursively explore this branch
            sub_results = generate_thoughts(new_state, depth - 1)
            results.extend(sub_results)

        return results

    # Get all paths and return the best
    all_paths = generate_thoughts(problem, max_depth)
    best_path = max(all_paths, key=lambda x: x[0])
    return best_path[1]
```

### Breadth vs. Depth

ToT can explore:
- **Breadth-first**: Generate all possibilities at each level, prune bad ones
- **Depth-first**: Follow promising paths deeply, backtrack on failure
- **Best-first**: Always expand the most promising node

```python
# Breadth-first with pruning
def tot_bfs(problem, width=5, depth=3):
    states = [problem]

    for d in range(depth):
        all_next = []
        for state in states:
            next_thoughts = generate_next_thoughts(state, n=width)
            all_next.extend(next_thoughts)

        # Keep only top-k promising states
        scores = [evaluate_promise(s) for s in all_next]
        states = select_top_k(all_next, scores, k=width)

    return max(states, key=evaluate_solution)
```

## Self-Consistency

### The Ensemble Approach

Instead of generating one answer, generate many with some randomness, then take the most common result:

```python
def self_consistent_answer(prompt, n_samples=10, temperature=0.7):
    """Generate multiple answers and return the consensus."""
    answers = []

    for _ in range(n_samples):
        # Higher temperature adds diversity
        response = llm.generate(prompt, temperature=temperature)
        answer = extract_final_answer(response)
        answers.append(answer)

    # Majority vote
    from collections import Counter
    answer_counts = Counter(answers)
    most_common = answer_counts.most_common(1)[0][0]

    # Optionally return confidence too
    confidence = answer_counts[most_common] / n_samples
    return most_common, confidence

# Usage
answer, conf = self_consistent_answer(
    "Q: What is 23 × 47? Let's think step by step.",
    n_samples=5
)
# Even if one path has an arithmetic error, majority gets it right
```

### Why It Works

- Different reasoning paths may reach the same correct answer
- Errors tend to be random, not systematic
- Majority voting filters out occasional mistakes
- Higher confidence when paths agree

### Trade-offs

```python
# Trade-off: accuracy vs. cost
n_samples = 1    # Fast, cheap, but may be wrong
n_samples = 5    # Good balance for most tasks
n_samples = 20   # High accuracy for critical tasks, expensive

# Temperature affects diversity
temp = 0.0    # All samples identical (no benefit)
temp = 0.5    # Moderate diversity
temp = 1.0    # High diversity, maybe too random
temp = 0.7    # Often good balance
```

## Structured Output Techniques

### JSON Mode

Many applications need structured data, not free-form text:

```python
# Prompt for structured output
prompt = """
Extract the following information from the text and return as JSON:
- person_name: string
- company: string
- job_title: string
- email: string or null

Text: "Please reach out to John Smith, VP of Engineering at
TechCorp (john.smith@techcorp.com) for technical questions."

JSON:
"""

# Response:
{
  "person_name": "John Smith",
  "job_title": "VP of Engineering",
  "company": "TechCorp",
  "email": "john.smith@techcorp.com"
}
```

### Schema Enforcement

Some APIs support schema enforcement:

```python
# OpenAI function calling / structured outputs
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract person info from..."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    }
)
# Guaranteed to return valid JSON matching schema
```

### Constrained Generation

For specific formats, constrain the generation:

```python
# Multiple choice - force specific output
prompt = """
Is this sentence grammatically correct?
Sentence: "Me and him went to store."

Answer with only the letter of the correct option:
A) Yes, it is correct
B) No, it is incorrect

Answer:"""

# Model should output just "B"
```

## Prompt Chaining

### Sequential Prompts

Break complex tasks into sequential steps:

```python
def analyze_document(document):
    # Step 1: Extract key points
    key_points = llm.generate(f"""
    Extract the 5 main points from this document:
    {document}
    Return as a numbered list.
    """)

    # Step 2: Analyze each point
    analyses = []
    for point in parse_list(key_points):
        analysis = llm.generate(f"""
        Provide a brief analysis of this point:
        {point}
        Consider: evidence strength, implications, limitations
        """)
        analyses.append(analysis)

    # Step 3: Synthesize
    synthesis = llm.generate(f"""
    Given these analyses:
    {format_analyses(analyses)}

    Write a cohesive executive summary in 3 paragraphs.
    """)

    return synthesis
```

### Parallel Prompts

Independent subtasks can run in parallel:

```python
import asyncio

async def multi_perspective_analysis(topic):
    # Run analyses in parallel
    perspectives = await asyncio.gather(
        llm.generate(f"Analyze {topic} from an economic perspective"),
        llm.generate(f"Analyze {topic} from a social perspective"),
        llm.generate(f"Analyze {topic} from an environmental perspective"),
    )

    # Synthesize
    synthesis = await llm.generate(f"""
    Synthesize these perspectives:
    Economic: {perspectives[0]}
    Social: {perspectives[1]}
    Environmental: {perspectives[2]}

    Identify tensions and synergies.
    """)

    return synthesis
```

## Least-to-Most Prompting

### Decomposing Complex Problems

First have the model decompose the problem, then solve each part:

```python
# Step 1: Decomposition
decomp_prompt = """
To solve this problem, what simpler sub-problems must be solved first?

Problem: Calculate the total cost of a 3-day trip to Paris including
flights, hotel, and daily expenses.

Sub-problems:
"""
# Model: "1. Find flight costs 2. Find hotel cost per night 3. Estimate daily expenses"

# Step 2: Solve each sub-problem
# Step 3: Combine solutions
```

### Example Trace

```
Problem: How long would it take to fill an Olympic swimming pool
with a garden hose?

Decomposition:
1. What's the volume of an Olympic swimming pool?
2. What's the flow rate of a garden hose?
3. Calculate time = volume / flow rate

Sub-problem 1: Olympic pool volume
An Olympic pool is 50m × 25m × 2m = 2,500 cubic meters
= 2,500,000 liters

Sub-problem 2: Garden hose flow rate
A typical garden hose: ~50 liters per minute
= 3,000 liters per hour
= 72,000 liters per day

Sub-problem 3: Calculate time
Time = 2,500,000 / 72,000 ≈ 34.7 days

Answer: Approximately 35 days running continuously.
```

## Key Takeaways

1. **ReAct combines reasoning with actions**, grounding LLM thinking in real information and tools.

2. **Tree-of-Thoughts explores multiple paths**, enabling backtracking and comparison of different approaches.

3. **Self-consistency uses ensemble voting** to improve reliability on reasoning tasks.

4. **Structured output techniques** ensure LLMs return data in usable formats.

5. **Prompt chaining** breaks complex tasks into manageable sequential or parallel steps.

6. **Least-to-most prompting** explicitly decomposes problems before solving.

## Further Reading

- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
- "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)
- "Least-to-Most Prompting Enables Complex Reasoning" (Zhou et al., 2022)
- "Language Model Cascades" (Dohan et al., 2022)
