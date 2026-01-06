# Current Applications and Limitations of AI Agents

## Introduction

AI agents have moved from research papers to real products. Coding assistants help developers write and debug software. Research agents gather and synthesize information. Automation agents handle complex workflows. But alongside genuine capabilities come significant limitations—agents that get stuck, make costly errors, or fail in subtle ways.

In this lesson, we'll survey the current landscape of AI agent applications, understand where they excel and struggle, and develop a realistic picture of what today's agents can and cannot reliably do.

## Coding Assistants

### The State of the Art

Coding assistants represent one of the most successful agent applications:

**GitHub Copilot**: Suggests code completions as you type
```python
# You start typing...
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""

# Copilot suggests...
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```

**Cursor**: AI-first code editor with multi-file editing and codebase understanding
```
User: "Refactor this authentication module to use JWT instead of sessions"
Cursor: [Analyzes auth module, proposes changes across 5 files, explains reasoning]
```

**Claude Code / Aider**: Command-line coding agents
```bash
$ aider "Add user authentication to the Flask app"
# Agent reads codebase, creates auth module, modifies routes, updates requirements
```

### What Coding Agents Do Well

- **Code completion**: Suggesting next lines or blocks
- **Explanation**: Describing what code does
- **Simple implementations**: Writing straightforward functions
- **Test generation**: Creating unit tests from function signatures
- **Refactoring**: Applying known patterns (rename variable, extract function)
- **Bug fixing**: Identifying and fixing common errors

### Current Limitations

```python
# Agents struggle with:

# 1. Complex architecture decisions
"Design the data layer for this distributed system"
# Agent may suggest something that works but has scaling issues

# 2. Subtle bugs with non-local effects
def process_order(order):
    # Agent might not realize this breaks
    # rate limiting middleware elsewhere
    order.timestamp = None  # Looks innocent...

# 3. Domain-specific knowledge
"Implement HIPAA-compliant data handling"
# Agent lacks deep regulatory knowledge

# 4. Debugging complex state issues
"Why does this work in dev but fail in prod?"
# Requires understanding environment differences, logs, timing
```

## Research and Analysis Agents

### Information Gathering

Research agents can synthesize information from multiple sources:

```python
# Task: Research competitive landscape for a startup

Agent workflow:
1. Search for competitors in the space
2. Visit each competitor's website
3. Extract key information (pricing, features, positioning)
4. Search for recent news and funding
5. Analyze market trends
6. Compile into structured report
```

### Real Applications

**Perplexity AI**: Answers questions with real-time search
```
User: "What's the latest research on long-context LLMs?"
Perplexity: [Searches academic papers, synthesizes findings, provides citations]
```

**Elicit**: Research assistant for literature review
```
User: [Uploads research question]
Elicit: [Finds relevant papers, extracts key claims, identifies methodology]
```

### Capabilities and Limits

**Works well**:
- Finding publicly available information
- Summarizing articles and papers
- Identifying key themes across sources
- Basic fact-checking against sources

**Struggles with**:
- Assessing source reliability and bias
- Recognizing outdated information
- Understanding context requiring deep expertise
- Verifying claims that require original research

## Business Automation Agents

### Workflow Automation

Agents can automate multi-step business processes:

```python
# Invoice processing workflow
trigger: "new_email_received"
agent_steps:
  - check_if_invoice()
  - extract_invoice_data()
  - validate_against_purchase_orders()
  - route_for_approval()
  - process_payment_if_approved()
  - update_accounting_system()
  - send_confirmation_email()
```

### Real Applications

**Customer Service**: Handle routine inquiries and escalate complex issues
```
Customer: "I need to change my shipping address for order #12345"
Agent: [Verifies identity, updates address, confirms with customer]
```

**Data Entry**: Process forms and update systems
```
Agent monitors:
  - Incoming application forms
  - Extracts relevant fields
  - Validates data
  - Updates CRM
  - Flags exceptions for human review
```

**Scheduling**: Coordinate meetings and appointments
```
User: "Find a time for a team meeting next week, avoid conflicting with the board meeting"
Agent: [Checks multiple calendars, proposes times, sends invites]
```

### Critical Limitations

```python
# Business automation failure modes:

# 1. Edge cases in business logic
# Agent approves expense that technically violates policy
# (the policy has implicit exceptions the agent doesn't know)

# 2. Integration failures
# Agent updates CRM but API call to billing system fails silently
# Data inconsistency propagates before detection

# 3. Missing context
# Agent follows procedure but situation requires judgment
# "This is a VIP customer, handle differently"

# 4. Cascading errors
# Small mistake early in workflow compounds into major issue
# Agent confidently continues executing incorrect plan
```

## Personal Assistants

### Current Products

**AI assistants** can help with personal tasks:

```
User: "Help me plan a trip to Japan in April"
Assistant:
  - Researches best regions for cherry blossom season
  - Suggests itinerary based on interests
  - Finds flight options
  - Recommends hotels
  - Provides packing suggestions
  - Creates day-by-day plan
```

### What Works

- Information gathering and synthesis
- Generating options and suggestions
- Draft creation (emails, messages, documents)
- Scheduling and reminders
- Basic research and summarization

### What Doesn't (Yet)

```python
# Personal assistant limitations:

# 1. Can't actually do things
# "Book that flight" → Still need human to click and pay

# 2. No persistent memory by default
# Each conversation starts fresh without context

# 3. Don't know your preferences deeply
# Suggests things that don't match your style

# 4. Can't access personal data easily
# Your calendar, email, files need explicit integration

# 5. Trust issues with autonomous action
# Would you let an agent send emails as you?
```

## Honest Assessment of Limitations

### The Reliability Gap

The fundamental challenge: agents are not reliably correct.

```python
# Success rates (illustrative, task-dependent):
task_success_rates = {
    "Simple code completion": "85-95%",
    "Complex multi-file refactoring": "40-60%",
    "Research summarization": "70-85%",
    "Autonomous multi-step tasks": "30-50%",
    "Critical business decisions": "Not recommended"
}

# The last 10-20% of reliability is the hardest
# And for many applications, 90% isn't enough
```

### Failure Modes

**Hallucination under uncertainty**:
```
Agent: "According to the company's Q3 2024 report..."
Reality: No such report exists; agent fabricated plausible-sounding citation
```

**Goal drift**:
```
User: "Research competitors and summarize in a table"
Agent: [Gets distracted by interesting tangent, spends tokens on irrelevant depth]
```

**Stuck in loops**:
```
Agent: search("topic") → no results
Agent: search("topic") → no results  # Tries same thing again
Agent: search("topic") → ...
```

**Compound errors**:
```
Step 1: Minor misunderstanding of task (90% correct)
Step 2: Based on step 1, makes reasonable choice (90%)
Step 3: Based on step 2... (90%)
...
Final: 0.9^10 = 35% chance of overall success
```

### When to Use (and Not Use) Agents

**Good use cases**:
- Human-in-the-loop for critical decisions
- Tasks where mistakes are cheap to fix
- First drafts that humans will review
- Information gathering to inform human decisions
- Automation of tedious but low-stakes tasks

**Risky use cases**:
- Fully autonomous critical operations
- Financial transactions without oversight
- Irreversible actions (deleting, sending)
- Security-sensitive decisions
- Medical, legal, or safety-critical domains

```python
# A reasonable framework:
def should_automate(task):
    reversible = is_easily_reversible(task)
    low_stakes = cost_of_error(task) < THRESHOLD
    human_review = has_human_checkpoint(task)

    if reversible and low_stakes:
        return "automate fully"
    elif human_review:
        return "automate with oversight"
    else:
        return "assist human, don't automate"
```

## The Path Forward

### Improving Reliability

Current research and engineering efforts:

```python
improvements = [
    "Better planning and decomposition",
    "Self-verification and critique",
    "Retrieval grounding (RAG)",
    "Tool reliability improvements",
    "Multi-agent verification",
    "Human oversight integration",
    "Better uncertainty quantification",
]
```

### Realistic Expectations

The technology is genuinely useful but not magic:

```
Think of agents as:
✓ Very capable interns who need supervision
✓ Tireless assistants for routine tasks
✓ Helpful tools that augment human capability

Not as:
✗ Perfect autonomous workers
✗ Replacements for human judgment
✗ Systems that can be trusted without oversight
```

## Key Takeaways

1. **Coding assistants are among the most successful agent applications**, excelling at completion, explanation, and routine tasks.

2. **Research agents can gather and synthesize information**, but struggle with deep expertise and source reliability.

3. **Business automation works best for structured, repeatable tasks** with clear error handling and human oversight.

4. **Reliability remains the core challenge**—agents are not consistently correct enough for autonomous critical operations.

5. **Success requires appropriate task selection**: low-stakes, reversible, or human-supervised tasks work best.

6. **The technology is evolving rapidly**, but honest assessment of current limitations is essential for responsible deployment.

## Further Reading

- "Challenges and Applications of Large Language Models" (Kaddour et al., 2023)
- "Large Language Models as Tool Makers" (Cai et al., 2023)
- "AI Agents That Matter" (Kapoor et al., 2024)
- GitHub Copilot research publications
- Case studies from LangChain, LlamaIndex, and AutoGen communities
