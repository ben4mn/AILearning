# Prompt Engineering Basics

## Introduction

Large language models are remarkably capable—but unlocking that capability requires knowing how to communicate with them. The art and science of crafting effective prompts has become a crucial skill in the AI era, spawning an entirely new discipline: **prompt engineering**.

Prompt engineering isn't just about typing questions. It's about understanding how LLMs process text, what patterns they've learned, and how to frame requests so that the model's training works in your favor. A well-crafted prompt can be the difference between a useless response and a brilliant one.

In this lesson, we'll explore what prompts are, why they matter so much, and the fundamental principles that guide effective prompt design.

## What Is a Prompt?

A prompt is any text input given to a language model. It's the starting point from which the model generates its response. Prompts can be:

**Simple questions**:
```
What is the capital of France?
```

**Complex instructions**:
```
Write a professional email to a client explaining that their
project deadline will be extended by two weeks due to
unexpected technical challenges. Be apologetic but confident.
```

**Rich contexts with examples**:
```
Classify the sentiment of movie reviews as positive or negative.

Review: "This film was absolutely fantastic! Best I've seen all year."
Sentiment: Positive

Review: "Terrible waste of time. The plot made no sense."
Sentiment: Negative

Review: "The acting was superb and the cinematography breathtaking."
Sentiment:
```

The model treats all of this as context for what comes next—and generates a continuation.

## Why Prompts Matter So Much

### The Same Model, Different Results

The exact same model can produce vastly different outputs depending on how you ask:

```python
# Weak prompt
prompt_1 = "Python sorting"
# Model might output anything—an explanation, code, history...

# Better prompt
prompt_2 = "Write a Python function that sorts a list of integers in descending order"
# Clear task, clear expected output

# Best prompt
prompt_3 = """
Write a Python function that:
- Takes a list of integers as input
- Returns the list sorted in descending order
- Handles empty lists gracefully
- Include a docstring and one example usage

Use type hints.
"""
# Specific requirements produce specific results
```

The model hasn't changed—only your ability to direct it has.

### LLMs as Text Completion Engines

Remember: LLMs are fundamentally trained to predict what comes next. This means:

```
Your prompt sets up expectations about what should follow.

Prompt: "Once upon a time"
→ Model expects: fairy tale continuation

Prompt: "def calculate_tax(income):"
→ Model expects: Python function body

Prompt: "The evidence suggests that"
→ Model expects: analytical conclusion
```

Understanding this helps explain why certain phrasings work better than others.

## Core Principles of Prompt Design

### 1. Be Specific

Vague prompts get vague answers. Specific prompts get useful responses.

```
# Vague
"Tell me about climate change"

# Specific
"Explain three key mechanisms by which greenhouse gases contribute
to global warming, suitable for a high school audience"

# Even more specific
"In 200 words, explain the greenhouse effect. Use an analogy
involving a car in sunlight. End with one actionable tip for
reducing personal carbon footprint."
```

### 2. Provide Context

Models have no memory beyond the current conversation. Include relevant context:

```python
# Without context - model must guess
prompt = "Should I accept the offer?"

# With context - model can reason
prompt = """
I'm a software engineer with 5 years of experience, currently
earning $120,000 annually. I've received a job offer for $145,000
at a startup with good growth potential but uncertain funding.

Should I accept the offer? What factors should I consider?
"""
```

### 3. Specify the Output Format

Want a list? Ask for a list. Want JSON? Specify JSON.

```python
# Unformatted output (harder to parse)
prompt = "What are the planets in our solar system?"

# Formatted output
prompt = """
List all planets in our solar system.
Format: Return as a numbered list, one planet per line,
ordered by distance from the sun.
"""

# Structured output
prompt = """
List all planets in our solar system.
Return as JSON with this structure:
{
  "planets": [
    {"name": "...", "type": "rocky|gas", "has_rings": true|false}
  ]
}
"""
```

### 4. Set the Tone and Persona

The same information can be delivered many ways:

```python
# Technical audience
prompt = "Explain gradient descent for a machine learning researcher"

# General audience
prompt = "Explain gradient descent as if I'm a smart 12-year-old"

# Specific persona
prompt = """
You are a patient and encouraging math tutor.
Explain gradient descent to someone who's struggling with calculus.
Use simple language and lots of analogies.
"""
```

### 5. Provide Examples

When possible, show what you want:

```python
prompt = """
Convert these sentences to past tense:

Input: "She walks to school."
Output: "She walked to school."

Input: "They eat lunch."
Output: "They ate lunch."

Input: "He runs fast."
Output:
"""
# Model understands the pattern and applies it
```

This technique, called **few-shot prompting**, is extremely powerful.

## The Anatomy of a Good Prompt

Let's dissect an effective prompt:

```
┌─────────────────────────────────────────────────────────────┐
│ CONTEXT                                                      │
│ You are an experienced technical writer specializing in      │
│ API documentation.                                           │
├─────────────────────────────────────────────────────────────┤
│ TASK                                                         │
│ Write documentation for the following Python function.       │
├─────────────────────────────────────────────────────────────┤
│ INPUT                                                        │
│ def fetch_user(user_id: int, include_posts: bool = False)   │
│     -> User:                                                 │
│     """Retrieve user from database."""                       │
│     ...                                                      │
├─────────────────────────────────────────────────────────────┤
│ REQUIREMENTS                                                 │
│ - Include a description, parameters section, and examples   │
│ - Use standard documentation format                          │
│ - Mention possible exceptions                                │
├─────────────────────────────────────────────────────────────┤
│ FORMAT                                                       │
│ Output as Markdown suitable for a documentation site.        │
└─────────────────────────────────────────────────────────────┘
```

This structure—context, task, input, requirements, format—works for most complex prompts.

## Common Prompt Patterns

### The Direct Instruction

Simply tell the model what to do:
```
Summarize the following article in three bullet points:
[article text]
```

### The Question

Ask directly:
```
What are the main differences between REST and GraphQL APIs?
```

### The Completion Prompt

Set up text for the model to continue:
```
The three most important factors in successful project management are:
1.
```

### The Critique/Analysis

Ask for evaluation:
```
Review this code for potential bugs and suggest improvements:
[code block]
```

### The Transformation

Convert input to output:
```
Translate this technical jargon into plain English:
"The API leverages RESTful paradigms with OAuth2 authentication..."
```

## What Doesn't Work

### Ambiguous Instructions

```
# Bad: Ambiguous
"Make this better"

# Good: Specific
"Improve this email by: making the tone more professional,
fixing grammatical errors, and adding a clear call to action"
```

### Assuming Context

```
# Bad: Assumes model knows your situation
"Should I go with the first option?"

# Good: Provides context
"I'm choosing between Option A ($50/month, basic features) and
Option B ($100/month, advanced features). My budget is tight but
I need the advanced analytics in Option B. What would you recommend?"
```

### Conflicting Instructions

```
# Bad: Contradictory
"Be extremely detailed but keep it under 50 words"

# Good: Prioritized
"Provide a concise summary (50-75 words) focusing on the
key conclusion. I can ask follow-up questions for details."
```

### Expecting Perfection First Try

Prompting is often iterative. Your first attempt may not be perfect:

```
Attempt 1: "Write marketing copy for a new app"
Result: Generic, not quite right

Attempt 2: "Write marketing copy for a productivity app
           targeting busy professionals. Emphasize time savings."
Result: Better, but too formal

Attempt 3: "Write casual, energetic marketing copy for a
           productivity app targeting busy professionals.
           Emphasize time savings. Use short punchy sentences."
Result: Much better!
```

## Iterative Refinement

The best prompts often emerge through iteration:

1. **Start simple**: Get baseline output
2. **Identify problems**: What's wrong or missing?
3. **Add constraints**: Address each issue
4. **Test variations**: See what works
5. **Refine further**: Polish for optimal results

```python
# Iteration example
v1 = "Write a poem about AI"
# Result: Generic, cliché

v2 = "Write a poem about AI from the perspective of the AI itself"
# Result: More interesting, but too long

v3 = "Write a short poem (8 lines) about AI from the AI's perspective,
      in the style of Emily Dickinson"
# Result: More focused, distinctive voice

v4 = "Write a short poem (8 lines, AABB rhyme scheme) about AI
      from the AI's perspective, contemplating consciousness,
      in the style of Emily Dickinson's contemplative poems"
# Result: Refined, specific, memorable
```

## Key Takeaways

1. **Prompts are the interface to LLMs**—they determine what you get out of these powerful systems.

2. **Be specific**: Vague prompts yield vague results. Define exactly what you want.

3. **Provide context**: The model knows nothing about your situation unless you tell it.

4. **Specify format**: Tell the model how to structure its response.

5. **Use examples**: Showing what you want is often clearer than describing it.

6. **Iterate**: Great prompts usually emerge through refinement, not on the first try.

## Further Reading

- "Prompt Engineering Guide" (DAIR.AI) - Comprehensive community resource
- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3 paper introducing few-shot prompting
- OpenAI's Prompt Engineering Best Practices documentation
- "A Survey of Prompting Methods" (Liu et al., 2023)
