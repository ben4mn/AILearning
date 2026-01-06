# Agent Architectures: ReAct, Plan-and-Execute, and Multi-Agent Systems

## Introduction

Not all agents are structured the same way. As the field has evolved, researchers and practitioners have developed distinct architectural patterns, each with its own strengths. Some agents reason and act in tight loops; others create detailed plans before execution; still others consist of multiple specialized agents collaborating.

In this lesson, we'll explore the major agent architectures: the ReAct pattern that interleaves reasoning with action, plan-and-execute approaches that separate planning from doing, and multi-agent systems where specialized agents collaborate on complex tasks.

## ReAct: Reasoning and Acting

### The ReAct Pattern

ReAct (Reasoning + Acting) is perhaps the most influential agent architecture. It interleaves thinking and doing in a tight loop:

```
Question: Who was the 2023 NBA Finals MVP and what team do they play for?

Thought 1: I need to find out who won the 2023 NBA Finals MVP.
Action 1: search("2023 NBA Finals MVP")
Observation 1: Nikola Jokic won the 2023 NBA Finals MVP.

Thought 2: Now I know the MVP. I need to find his team.
Action 2: search("Nikola Jokic team 2023")
Observation 2: Nikola Jokic plays for the Denver Nuggets.

Thought 3: I have all the information needed.
Answer: Nikola Jokic was the 2023 NBA Finals MVP, and he plays for the Denver Nuggets.
```

### Implementation

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.prompt_template = """
Answer the question using the available tools.

Tools:
{tool_descriptions}

Format:
Thought: [reasoning about what to do]
Action: [tool_name(arguments)]
Observation: [result from tool]
... (repeat as needed)
Answer: [final answer]

Question: {question}

{history}"""

    def run(self, question, max_iterations=10):
        history = ""

        for i in range(max_iterations):
            # Generate next thought/action
            prompt = self.prompt_template.format(
                tool_descriptions=self.format_tools(),
                question=question,
                history=history
            )

            response = self.llm.generate(prompt, stop=["Observation:"])

            # Check if we have a final answer
            if "Answer:" in response:
                return self.extract_answer(response)

            # Parse and execute action
            if "Action:" in response:
                action, args = self.parse_action(response)
                observation = self.execute(action, args)
                history += f"{response}\nObservation: {observation}\n\n"

        return "Could not find answer within iteration limit"

    def execute(self, action_name, arguments):
        if action_name in self.tools:
            return self.tools[action_name](**arguments)
        return f"Unknown tool: {action_name}"
```

### Strengths and Weaknesses

**Strengths**:
- Naturally corrects course based on observations
- Grounded in real information
- Interpretable reasoning trace
- Simple to implement

**Weaknesses**:
- Can be token-expensive (reasoning at every step)
- May get stuck in loops
- No explicit long-term planning
- Difficulty with complex multi-step tasks

## Plan-and-Execute

### Separating Planning from Execution

The plan-and-execute pattern first creates a comprehensive plan, then executes steps systematically:

```
Task: Research and write a blog post about renewable energy trends

PLANNING PHASE:
1. Search for recent statistics on renewable energy adoption
2. Find information about solar energy trends
3. Find information about wind energy trends
4. Identify emerging technologies in renewables
5. Compile findings into structured notes
6. Write introduction section
7. Write body sections for each energy type
8. Write conclusion with future outlook
9. Review and polish the post

EXECUTION PHASE:
[Executing step 1: search("renewable energy adoption statistics 2024")]
[Result: Global renewable capacity grew 50% in 2023...]

[Executing step 2: search("solar energy trends 2024")]
[Result: Solar installations reached record levels...]

[Executing step 3: ...]
...
```

### Implementation

```python
class PlanAndExecuteAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.planner_prompt = """
Create a step-by-step plan to accomplish this task.
Each step should be concrete and actionable.

Available tools: {tools}

Task: {task}

Plan (numbered list):"""

        self.executor_prompt = """
Execute this step of the plan.

Available tools: {tools}
Task context: {context}
Previous results: {previous_results}

Current step: {step}

Think about what tool to use and execute it.
Thought:
Action:"""

    def run(self, task):
        # Phase 1: Create plan
        plan = self.create_plan(task)
        print(f"Plan created with {len(plan)} steps")

        # Phase 2: Execute plan
        results = []
        context = task

        for i, step in enumerate(plan):
            print(f"Executing step {i+1}: {step}")

            result = self.execute_step(step, context, results)
            results.append({"step": step, "result": result})

            # Optional: Check if plan needs revision
            if self.needs_replanning(result):
                plan = self.revise_plan(task, results, plan[i+1:])

        # Phase 3: Synthesize final result
        return self.synthesize(task, results)

    def create_plan(self, task):
        prompt = self.planner_prompt.format(
            tools=self.format_tools(),
            task=task
        )
        response = self.llm.generate(prompt)
        return self.parse_plan(response)

    def execute_step(self, step, context, previous_results):
        prompt = self.executor_prompt.format(
            tools=self.format_tools(),
            context=context,
            previous_results=self.format_results(previous_results),
            step=step
        )
        response = self.llm.generate(prompt, stop=["Observation:"])

        if "Action:" in response:
            action, args = self.parse_action(response)
            return self.execute(action, args)

        return response
```

### Adaptive Replanning

Real-world execution often requires plan adjustments:

```python
def needs_replanning(self, result):
    """Check if the plan needs to be revised."""
    # Error occurred
    if "error" in str(result).lower():
        return True

    # Unexpected result changes the situation
    if self.is_significant_deviation(result):
        return True

    return False

def revise_plan(self, original_task, completed_results, remaining_steps):
    """Revise the plan based on current progress."""
    prompt = f"""
The original task was: {original_task}

Completed steps and results:
{self.format_results(completed_results)}

The remaining plan was:
{remaining_steps}

Based on the results so far, revise the remaining plan if needed.
The plan should still accomplish the original task.

Revised remaining steps:"""

    response = self.llm.generate(prompt)
    return self.parse_plan(response)
```

### Strengths and Weaknesses

**Strengths**:
- Better for complex, multi-step tasks
- Can handle longer horizons
- More structured and predictable
- Easier to monitor progress

**Weaknesses**:
- Planning may not anticipate all contingencies
- Less flexible to unexpected observations
- May waste effort if early steps invalidate later plans
- Two-phase approach adds latency

## Multi-Agent Systems

### Why Multiple Agents?

Some tasks benefit from specialized roles:

```
Task: Review a codebase for security vulnerabilities and write a report

Agent 1 (Security Analyst):
- Specialized in finding vulnerabilities
- Knows common attack patterns
- Focuses on code analysis

Agent 2 (Code Context Expert):
- Understands the codebase architecture
- Provides context about how code is used
- Identifies high-risk areas

Agent 3 (Technical Writer):
- Synthesizes findings into clear reports
- Prioritizes issues for the audience
- Creates actionable recommendations
```

### Basic Multi-Agent Architecture

```python
class MultiAgentSystem:
    def __init__(self, agents, coordinator):
        self.agents = agents
        self.coordinator = coordinator

    def run(self, task):
        # Coordinator breaks down task
        subtasks = self.coordinator.decompose(task)

        # Assign subtasks to specialists
        results = {}
        for subtask in subtasks:
            agent = self.coordinator.assign(subtask, self.agents)
            result = agent.execute(subtask)
            results[subtask.id] = result

        # Synthesize results
        return self.coordinator.synthesize(task, results)


class CoordinatorAgent:
    def decompose(self, task):
        """Break task into subtasks for specialists."""
        prompt = f"""
Break this task into subtasks for specialist agents.

Available specialists:
{self.describe_agents()}

Task: {task}

Subtasks (with assigned specialist):"""
        # ... generate and parse subtasks

    def assign(self, subtask, agents):
        """Select the best agent for a subtask."""
        return agents[subtask.assigned_to]

    def synthesize(self, task, results):
        """Combine specialist outputs into final result."""
        prompt = f"""
Combine these specialist outputs into a coherent result.

Original task: {task}

Specialist outputs:
{self.format_results(results)}

Synthesized result:"""
        return self.llm.generate(prompt)
```

### Agent Communication

Agents may need to communicate:

```python
class CollaborativeAgents:
    def __init__(self):
        self.message_queue = []
        self.agents = {}

    def send_message(self, from_agent, to_agent, message):
        """Send a message between agents."""
        self.message_queue.append({
            "from": from_agent,
            "to": to_agent,
            "content": message,
            "timestamp": time.time()
        })

    def get_messages(self, agent_name):
        """Get messages for an agent."""
        return [m for m in self.message_queue if m["to"] == agent_name]

    def collaborative_loop(self, task, max_rounds=5):
        """Agents work together in rounds."""
        for round in range(max_rounds):
            for name, agent in self.agents.items():
                # Agent receives messages
                messages = self.get_messages(name)

                # Agent processes and potentially responds
                actions = agent.process(task, messages)

                for action in actions:
                    if action["type"] == "message":
                        self.send_message(name, action["to"], action["content"])
                    elif action["type"] == "complete":
                        return action["result"]

        return self.synthesize_current_state()
```

### Debate and Verification

Multi-agent debate can improve accuracy:

```python
def agent_debate(question, agents, max_rounds=3):
    """Multiple agents debate to reach a better answer."""
    answers = {}

    # Initial answers
    for agent in agents:
        answers[agent.name] = agent.answer(question)

    # Debate rounds
    for round in range(max_rounds):
        for agent in agents:
            other_answers = {
                name: ans for name, ans in answers.items()
                if name != agent.name
            }

            prompt = f"""
Question: {question}

Your previous answer: {answers[agent.name]}

Other agents' answers:
{format_dict(other_answers)}

Consider the other perspectives. Do you want to revise your answer?
If so, explain your reasoning.

Your answer:"""

            answers[agent.name] = agent.llm.generate(prompt)

    # Check for consensus or vote
    return determine_final_answer(answers)
```

### Specialized Agent Types

```python
# Researcher: Gathers information
class ResearchAgent:
    def __init__(self, search_tools):
        self.tools = search_tools

    def research(self, topic):
        # Systematically search and compile information
        ...

# Critic: Reviews and improves outputs
class CriticAgent:
    def critique(self, content, criteria):
        prompt = f"""
Review this content against the following criteria:
{criteria}

Content:
{content}

Provide specific, constructive feedback:"""
        return self.llm.generate(prompt)

# Executor: Carries out specific actions
class ExecutorAgent:
    def execute(self, action_plan):
        # Carefully execute each action
        ...

# Verifier: Checks correctness
class VerifierAgent:
    def verify(self, claim, evidence):
        prompt = f"""
Verify whether this claim is supported by the evidence.

Claim: {claim}
Evidence: {evidence}

Is the claim supported? Explain your reasoning."""
        return self.llm.generate(prompt)
```

## Choosing an Architecture

| Architecture | Best For | Complexity | Reliability |
|-------------|----------|------------|-------------|
| ReAct | Simple, linear tasks | Low | Medium |
| Plan-Execute | Complex, predictable tasks | Medium | Medium-High |
| Multi-Agent | Specialized, collaborative tasks | High | Varies |

### Decision Factors

**Use ReAct when**:
- Tasks are relatively simple
- You need quick iteration
- The path forward isn't clear upfront
- Token efficiency isn't critical

**Use Plan-and-Execute when**:
- Tasks have clear multi-step structure
- Monitoring progress is important
- Some steps are expensive to retry
- You want more predictability

**Use Multi-Agent when**:
- Task requires diverse expertise
- Verification/critique is valuable
- Parallelization is possible
- Building specialized, reusable components

## Key Takeaways

1. **ReAct interleaves reasoning and acting**, making it flexible but potentially expensive for long tasks.

2. **Plan-and-Execute separates planning from doing**, providing structure for complex tasks but requiring replanning mechanisms.

3. **Multi-agent systems enable specialization and collaboration**, powerful but more complex to orchestrate.

4. **Architecture choice depends on task characteristics**: complexity, predictability, need for specialization, and reliability requirements.

5. **Hybrid approaches often work best**: plan-then-ReAct, or multi-agent with ReAct individuals.

6. **All architectures benefit from good error handling and human oversight** for critical applications.

## Further Reading

- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
- "Plan-and-Solve Prompting" (Wang et al., 2023)
- "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (Wu et al., 2023)
- "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society" (Li et al., 2023)
- "Debate Improves Reasoning" (Du et al., 2023)
