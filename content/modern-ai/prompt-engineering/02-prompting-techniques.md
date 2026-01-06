# Prompting Techniques: Zero-Shot, Few-Shot, and Chain-of-Thought

## Introduction

As researchers and practitioners explored large language models, they discovered that *how* you ask matters as much as *what* you ask. Different prompting strategies can dramatically improve performance on various tasks—sometimes turning failure into success without changing the model at all.

In this lesson, we'll explore the foundational prompting techniques that have proven most effective: zero-shot prompting, few-shot learning, and chain-of-thought reasoning. These techniques form the basis of modern prompt engineering and are essential tools for anyone working with LLMs.

## Zero-Shot Prompting

### The Concept

Zero-shot prompting means asking the model to perform a task without providing any examples. You rely entirely on the model's pre-training and your instructions:

```
Classify the following text as either sports, politics, or entertainment:

"The Lakers won their game last night with a stunning 3-pointer in the
final seconds, sending fans into celebration."

Category:
```

The model has never seen this exact task during training, but it understands the concepts involved and can classify appropriately.

### When Zero-Shot Works Well

Zero-shot prompting excels when:
- The task is well-defined and common
- The model has relevant knowledge from pre-training
- The instructions are clear and unambiguous

```python
# Good zero-shot tasks
tasks = [
    "Translate this English text to Spanish: ...",  # Clear, common
    "Summarize this article in 3 sentences: ...",   # Well-defined
    "Is this review positive or negative? ...",     # Simple classification
    "Fix the grammar in this sentence: ...",        # Clear objective
]
```

### When Zero-Shot Struggles

Zero-shot can fail when:
- The task format is unusual or ambiguous
- The output format isn't obvious
- The task requires specialized knowledge or style

```python
# Zero-shot might struggle
prompt = "Rate this restaurant review on our custom scale"
# What scale? What criteria? Model must guess.

# Better: Add specificity (still zero-shot)
prompt = """
Rate this restaurant review on a scale of 1-5:
1 = Very Negative
2 = Negative
3 = Neutral
4 = Positive
5 = Very Positive

Review: [text]
Rating:
"""
```

## Few-Shot Prompting

### The Concept

Few-shot prompting provides examples of the task before asking the model to perform it. These examples demonstrate the desired pattern:

```
Convert the following to pig latin:

English: hello
Pig Latin: ellohay

English: world
Pig Latin: orldway

English: computer
Pig Latin: omputercay

English: programming
Pig Latin:
```

The model learns the pattern from examples and applies it to new input.

### The Magic of In-Context Learning

Few-shot prompting leverages **in-context learning**—the model's ability to adapt to new tasks based solely on examples in the prompt, without any weight updates:

```python
# No training required!
# The model "learns" the task from examples in the prompt

prompt = """
Translate company names to their stock tickers:

Company: Apple Inc.
Ticker: AAPL

Company: Microsoft Corporation
Ticker: MSFT

Company: Alphabet Inc.
Ticker: GOOGL

Company: Amazon.com Inc.
Ticker:
"""
# Model outputs: AMZN
```

This is remarkable—the model wasn't trained on this specific task, yet it picks up the pattern immediately.

### How Many Examples?

The number of examples matters:

| Shots | Use Case |
|-------|----------|
| 0 (zero-shot) | Simple, well-defined tasks |
| 1-2 (one/two-shot) | Clear patterns, common tasks |
| 3-5 (few-shot) | Moderate complexity, specific formats |
| 5-10+ (many-shot) | Complex patterns, unusual formats |

```python
# More examples help with unusual formats
weird_format_prompt = """
Convert sentences to our custom markup format:

Input: Hello world
Output: <<GREET>>Hello<</GREET>> <<NOUN>>world<</NOUN>>

Input: Good morning everyone
Output: <<GREET>>Good morning<</GREET>> <<NOUN>>everyone<</NOUN>>

Input: Hi there friend
Output: <<GREET>>Hi there<</GREET>> <<NOUN>>friend<</NOUN>>

Input: Welcome dear guest
Output:
"""
# With 3 examples, model understands the unusual format
```

### Best Practices for Few-Shot

**Choose representative examples**:
```python
# Bad: All examples too similar
examples = [
    ("cat", "animal"),
    ("dog", "animal"),
    ("bird", "animal"),
]

# Good: Cover the output space
examples = [
    ("cat", "animal"),
    ("apple", "fruit"),
    ("car", "vehicle"),
]
```

**Order can matter**:
```python
# Put most relevant examples near the end
# Recent context tends to have more influence
```

**Match difficulty level**:
```python
# If your test cases are complex, include complex examples
# Simple examples may not demonstrate handling of edge cases
```

## Chain-of-Thought Prompting

### The Breakthrough

In 2022, researchers discovered something remarkable: asking models to "think step by step" dramatically improved reasoning performance. This technique, called **Chain-of-Thought (CoT) prompting**, has become essential for complex tasks.

### Zero-Shot CoT

The simplest form just adds a magic phrase:

```python
# Without CoT
prompt = """
If there are 3 cars in a parking lot and 2 more cars arrive,
then 1 car leaves, how many cars are in the parking lot?
Answer:
"""
# Model might answer incorrectly

# With Zero-Shot CoT
prompt = """
If there are 3 cars in a parking lot and 2 more cars arrive,
then 1 car leaves, how many cars are in the parking lot?

Let's think step by step.
"""
# Model: "Start with 3 cars. 2 arrive: 3 + 2 = 5. 1 leaves: 5 - 1 = 4. Answer: 4"
```

### Few-Shot CoT

Even better: show examples of step-by-step reasoning:

```python
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 balls each is 2 * 3 = 6 balls.
5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
bought 6 more, how many apples do they have?

A: The cafeteria started with 23 apples. They used 20, leaving
23 - 20 = 3. They bought 6 more, so 3 + 6 = 9. The answer is 9.

Q: There are 15 trees in the grove. Grove workers will plant trees today.
After they are done, there will be 21 trees. How many trees did they plant?

A:
"""
# Model follows the demonstrated reasoning pattern
```

### Why Does CoT Work?

Several hypotheses:
1. **Breaks complex problems into steps**: Each step is easier than solving all at once
2. **Activates relevant knowledge**: Mentioning concepts brings related training into play
3. **Reduces error accumulation**: Explicit intermediate results can be checked
4. **Matches training data**: Step-by-step solutions appear in training data (textbooks, tutorials)

### When to Use CoT

Chain-of-thought is most valuable for:
- Mathematical reasoning
- Multi-step logic problems
- Complex decision-making
- Causal reasoning
- Any task where humans would "show their work"

```python
# Good candidates for CoT
tasks = [
    "Solve this word problem",
    "Debug this code and explain the issue",
    "Analyze the pros and cons of this decision",
    "Explain how you would approach this design problem",
]

# Less benefit from CoT
simple_tasks = [
    "Translate this sentence",
    "Classify this sentiment",
    "Extract the email address from this text",
]
```

## Combining Techniques

These techniques combine naturally:

### Few-Shot + CoT

Provide examples that include reasoning:

```python
prompt = """
Determine if the second sentence follows logically from the first.

Premise: All roses are flowers. Some flowers fade quickly.
Conclusion: Some roses fade quickly.
Analysis: The premise says all roses are flowers, but only SOME flowers
fade quickly. We can't conclude which flowers those are—they might not
include roses. This is invalid logic.
Answer: Does not follow

Premise: No reptiles have fur. All snakes are reptiles.
Conclusion: No snakes have fur.
Analysis: If no reptiles have fur, and all snakes are reptiles, then
snakes (being reptiles) have no fur. This is valid logic.
Answer: Follows

Premise: Some birds can fly. All eagles are birds.
Conclusion: All eagles can fly.
Analysis:
"""
```

### Zero-Shot + Explicit Instructions

Be very specific about the reasoning process:

```python
prompt = """
Analyze the following code for bugs.

Steps to follow:
1. Read through the code line by line
2. Identify any syntax errors
3. Check for logic errors
4. Consider edge cases
5. List all bugs found with line numbers
6. Suggest fixes for each bug

Code:
```python
def calculate_average(numbers):
    sum = 0
    for num in numbers:
        sum += num
    return sum / len(numbers)
```

Analysis:
"""
```

## Self-Consistency

An advanced technique: generate multiple chain-of-thought responses and take the majority answer:

```python
def self_consistent_answer(prompt, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = llm.generate(prompt, temperature=0.7)  # Some randomness
        answer = extract_final_answer(response)
        answers.append(answer)

    # Return most common answer
    return most_frequent(answers)

# Different reasoning paths may reach same correct answer
# Occasional errors get outvoted
```

## Practical Examples

### Example 1: Data Extraction

```python
# Zero-shot with format specification
prompt = """
Extract the following information from the text:
- Person's name
- Company
- Job title
- Email (if present)

Return as JSON.

Text: "John Smith is the CEO of TechCorp and can be reached at
jsmith@techcorp.com for business inquiries."

Extracted:
"""
```

### Example 2: Code Generation

```python
# Few-shot with code examples
prompt = """
Write a Python function based on the description.

Description: Check if a string is a palindrome
```python
def is_palindrome(s):
    clean = s.lower().replace(" ", "")
    return clean == clean[::-1]
```

Description: Count vowels in a string
```python
def count_vowels(s):
    return sum(1 for c in s.lower() if c in 'aeiou')
```

Description: Find the longest word in a sentence
```python
"""
```

### Example 3: Complex Reasoning

```python
# Few-shot CoT for logic
prompt = """
Determine if the argument is logically valid.

Argument: All mammals are warm-blooded. Whales are mammals.
Therefore, whales are warm-blooded.

Reasoning: This follows the pattern "All A are B. X is A. Therefore X is B."
This is valid syllogistic reasoning (Barbara form).
Valid: Yes

Argument: Some dogs are friendly. Rover is a dog.
Therefore, Rover is friendly.

Reasoning: "Some dogs are friendly" doesn't tell us about ALL dogs.
Rover could be in the friendly group or not. We can't conclude definitively.
Valid: No

Argument: If it rains, the ground gets wet. The ground is wet.
Therefore, it rained.

Reasoning:
"""
```

## Key Takeaways

1. **Zero-shot works for simple, well-defined tasks** where clear instructions suffice.

2. **Few-shot prompting enables in-context learning**—the model adapts to new tasks from examples alone.

3. **Chain-of-thought dramatically improves reasoning** by making the model "show its work."

4. **"Let's think step by step" is surprisingly powerful**—simple phrase, significant improvement.

5. **Techniques combine**: Few-shot examples with chain-of-thought reasoning often outperform either alone.

6. **Match the technique to the task**: Simple tasks don't need complex prompting; reasoning tasks benefit from CoT.

## Further Reading

- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
- "Large Language Models are Zero-Shot Reasoners" (Kojima et al., 2022)
- "Language Models are Few-Shot Learners" (Brown et al., 2020)
- "A Survey of Chain of Thought Reasoning" (Chu et al., 2023)
