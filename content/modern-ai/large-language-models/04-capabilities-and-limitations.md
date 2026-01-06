# Capabilities and Limitations of Large Language Models

## Introduction

Large language models can do remarkable things. They write code, compose poetry, explain quantum physics, and engage in nuanced conversation. Yet they also make confident errors, invent plausible-sounding falsehoods, and fail at tasks a child could handle. Understanding both sides—what LLMs can and cannot do—is essential for using them effectively and thinking clearly about their implications.

In this lesson, we'll explore the genuine capabilities that make LLMs valuable, the limitations that require caution, and the deeper questions these systems raise about intelligence and understanding. This balanced view will help you become a more effective user while maintaining appropriate skepticism.

## What LLMs Do Well

### Natural Language Understanding and Generation

LLMs excel at language tasks across an impressive range:

**Summarization**: Condensing long documents while preserving key points
```
Input: [5000-word research paper]
Prompt: "Summarize this paper in 3 bullet points for a non-technical audience"
Output: Clear, accurate summary accessible to general readers
```

**Translation**: Converting between languages with contextual awareness
```
Prompt: "Translate to French, maintaining the formal business tone"
Works well for major languages; quality varies for low-resource languages
```

**Style Transfer**: Rewriting text in different voices or formats
```
Prompt: "Rewrite this technical documentation as a casual blog post"
LLMs capture tone, vocabulary, and structural patterns effectively
```

**Content Generation**: Creating drafts, outlines, and first versions
```
Prompt: "Write a product description for a smart water bottle"
Produces fluent, relevant content that typically needs human editing
```

### Code Understanding and Generation

Programming is one of LLMs' strongest applications:

```python
# LLMs can:
# 1. Explain existing code
def complex_function(data):
    # LLM can describe what this does in plain English
    return [x for x in data if x > 0 and x % 2 == 0]

# 2. Generate code from descriptions
# Prompt: "Write a function to find all prime numbers up to n"
# LLM produces working implementation

# 3. Debug issues
# "This code raises IndexError on line 23. Here's the traceback..."
# LLM identifies the issue and suggests fixes

# 4. Translate between languages
# "Convert this Python to JavaScript"
```

### Information Synthesis

LLMs can combine information from their training:
- Comparing concepts across domains
- Explaining relationships between ideas
- Providing overviews of topics
- Answering factual questions (with caveats)

### Reasoning and Problem-Solving

With appropriate prompting, LLMs can work through multi-step problems:

```
Prompt: "A store has 100 apples. On Monday they sell 30% of their
apples. On Tuesday they receive a shipment of 50 apples. On
Wednesday they sell half of what they have. How many apples
remain? Think step by step."

LLM Response:
"Let me work through this step by step:
1. Starting apples: 100
2. Monday sales: 100 × 0.30 = 30 apples sold
   Remaining: 100 - 30 = 70 apples
3. Tuesday shipment: 70 + 50 = 120 apples
4. Wednesday sales: 120 ÷ 2 = 60 apples sold
   Remaining: 120 - 60 = 60 apples

The store has 60 apples remaining."
```

## What LLMs Struggle With

### Hallucinations: Confident Falsehoods

Perhaps the most significant limitation is **hallucination**—generating plausible-sounding but false information. LLMs don't distinguish between "things I learned from training data" and "things I'm making up":

```
User: "Who wrote the book 'The Silicon Mind' published in 2019?"

LLM: "The Silicon Mind was written by Dr. Patricia Hernandez,
a professor of cognitive science at MIT. It explores the
relationship between artificial intelligence and human
consciousness..."

Reality: This book may not exist. The author is invented.
The details are fabricated. But it all sounds convincing.
```

Hallucination happens because:
- The model optimizes for plausible-sounding text, not truth
- It has no mechanism to verify information
- It can't distinguish reliable from unreliable training data
- Generating "I don't know" is less natural than completing the pattern

### Mathematical Reasoning

Despite appearances, LLMs often struggle with arithmetic and mathematical reasoning:

```
Prompt: "What is 47 × 83?"

LLM might say: "3,881" (correct) or "3,901" (wrong)

The model doesn't "calculate"—it pattern-matches based on
similar problems seen during training. Novel problems may fail.
```

For reliable math, LLMs should use code execution:
```python
# Better approach
prompt = "Write Python code to calculate 47 * 83, then run it"
# LLM generates: print(47 * 83)
# Code interpreter returns: 3901
```

### Current Information

LLMs have a **knowledge cutoff**—they only know information from their training data:

```
User: "Who won the Super Bowl in 2025?"

LLM: "I don't have information about events after my
knowledge cutoff date of [X]. For current information,
please check recent news sources."
```

This limitation drives the development of RAG (Retrieval-Augmented Generation) and tool use, which we'll cover in later lessons.

### Consistent Long-Term State

LLMs process each request fresh. They don't truly "remember" previous conversations or maintain persistent state:

```
Conversation 1: "My name is Alex"
Conversation 2: "What's my name?"
LLM: [Cannot access Conversation 1]
```

Within a single conversation, they rely on the context window, which has limits. Very long conversations may lose early context.

### Spatial and Physical Reasoning

LLMs struggle with tasks requiring spatial understanding:

```
Prompt: "I'm facing north. I turn right, walk 3 blocks,
turn left, walk 2 blocks, turn around, walk 1 block.
Which direction am I now facing?"

LLMs often get this wrong because they don't have a
spatial model—just patterns of text about directions.
```

### Counting and Precise Operations

Tasks requiring exact counting often fail:

```
Prompt: "How many 'r's are in the word 'strawberry'?"

LLM: "There are 2 r's in strawberry"
Correct answer: 3

The model doesn't "see" letters individually—it processes
tokens that may not align with letter boundaries.
```

## The Deeper Questions

### Do LLMs "Understand"?

This is perhaps the most debated question in contemporary AI. Consider two perspectives:

**The skeptical view**: LLMs are sophisticated pattern matchers. They predict statistically likely text without any genuine comprehension. When GPT discusses quantum physics, it's remixing patterns from training data, not understanding physics.

**The nuanced view**: "Understanding" is poorly defined. LLMs exhibit functional understanding—they respond appropriately to context, make reasonable inferences, and adapt to novel situations. Whether there's "something it's like" to be an LLM remains unknown.

### The Chinese Room Revisited

Philosopher John Searle's Chinese Room thought experiment feels newly relevant:

> Imagine someone in a room following rules to manipulate Chinese symbols. They produce correct responses without understanding Chinese. Similarly, LLMs manipulate tokens without understanding meaning.

Counter-arguments:
- The system as a whole might understand, even if components don't
- Human brains are also "just" neurons following rules
- Functional behavior may be sufficient for understanding

### Stochastic Parrots or Something More?

In their influential 2021 paper, Bender, Gebru, et al. argued that LLMs are "stochastic parrots"—they reproduce patterns from training data without genuine comprehension.

However, LLMs also demonstrate:
- Novel combinations of ideas not explicit in training data
- Transfer across domains
- Apparent reasoning and inference

The truth likely lies somewhere between "mere pattern matching" and "genuine understanding"—a space our concepts may not yet have words for.

## Practical Implications

Understanding these capabilities and limitations suggests practical strategies:

### When to Trust LLMs

**Trust more for**:
- First drafts and brainstorming
- Explaining well-established concepts
- Code syntax and common patterns
- Summarizing provided text
- Style and formatting tasks

**Trust less for**:
- Specific facts and figures
- Recent events
- Precise calculations
- Legal, medical, or financial advice
- Claims about obscure topics

### Verification Strategies

```python
# Good practice: Cross-reference important claims
response = llm.generate("What is the population of Singapore?")

# Bad: Trust directly
# population = response

# Good: Verify
# 1. Check multiple sources
# 2. Use retrieval-augmented generation
# 3. For math, use code execution
# 4. For recent info, search the web
```

### Appropriate Use Cases

| Use Case | Suitability | Notes |
|----------|-------------|-------|
| Drafting emails | Excellent | Human review before sending |
| Coding assistance | Very good | Test generated code |
| Research summaries | Good | Verify claims |
| Medical diagnosis | Poor | Requires expert verification |
| Legal advice | Poor | Professional review essential |
| Creative brainstorming | Excellent | Inspiration, not final product |
| Tutoring explanations | Good | For well-established topics |

## The Capability Trajectory

It's worth noting that limitations today may not persist forever. LLMs have consistently improved at tasks once thought beyond reach:

- 2019: "LLMs can't do math" → Now: Reasonable with chain-of-thought
- 2020: "LLMs can't reason" → Now: Demonstrable multi-step reasoning
- 2021: "LLMs can't use tools" → Now: Tool use is standard

However, some limitations may prove fundamental:
- No access to real-time information without retrieval
- No genuine memory without explicit systems
- Potential for hallucination may be inherent to the architecture

## Key Takeaways

1. **LLMs excel at language tasks**: summarization, translation, style transfer, and fluent generation are genuine strengths.

2. **Hallucination is a fundamental limitation**: LLMs generate plausible text, not verified truth. Always verify important claims.

3. **Mathematical precision requires tools**: Don't trust LLMs for arithmetic—use code execution for reliable computation.

4. **Knowledge has temporal limits**: LLMs don't know about events after their training cutoff.

5. **The understanding question remains open**: Whether LLMs "truly" understand is philosophically contested and may not have a clear answer.

6. **Practical wisdom involves knowing when to trust**: Match tasks to capabilities and build verification into your workflow.

## Further Reading

- "On the Dangers of Stochastic Parrots" (Bender, Gebru et al., 2021)
- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3 paper discussing capabilities
- "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022)
- "Sparks of Artificial General Intelligence: Early experiments with GPT-4" (Microsoft Research, 2023)
- "Do Large Language Models Know What They Don't Know?" (Yin et al., 2023)
