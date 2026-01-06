# What Are Large Language Models?

## Introduction

If you've been following the AI revolution of the 2020s, you've undoubtedly encountered the term "Large Language Model" or LLM. These systems have captured the public imagination in a way that few technological advances ever have, sparking debates about everything from the future of work to the nature of intelligence itself. But what exactly *are* large language models, and why have they proven so transformative?

In this lesson, we'll explore the fundamental nature of LLMs, understand what makes them "large," and discover the surprising capabilities that emerge when language models reach sufficient scale. By the end, you'll have a solid foundation for understanding the technology that's reshaping how we interact with computers.

## The Basic Concept: Predicting the Next Token

At their core, large language models are neural networks trained on a deceptively simple task: given a sequence of text, predict what comes next. That's it. No explicit grammar rules, no hand-coded knowledge bases, no carefully curated ontologies. Just billions of examples of text and the objective of predicting the next piece.

This approach, called **autoregressive language modeling**, works token by token. A token is typically a word or a piece of a word (we'll explore tokenization in depth in the next topic). Given the tokens "The capital of France is," the model assigns probabilities to every possible next token in its vocabulary. "Paris" should get a high probability, while "elephant" should get a very low one.

```python
# Conceptual example (not actual implementation)
def generate_next_token(model, input_text):
    # Convert text to tokens
    tokens = tokenize(input_text)

    # Get probability distribution over vocabulary
    probabilities = model.forward(tokens)

    # Sample or select the most likely next token
    next_token = sample(probabilities)

    return next_token
```

What makes this seemingly simple objective so powerful is the *scale* at which it's applied. When you train on trillions of tokens from the internet, books, code, and scientific papers, predicting the next word requires learning an enormous amount about the world. To predict that "The Eiffel Tower is located in" should be followed by "Paris," the model must learn geography. To complete code correctly, it must learn programming. To continue a logical argument, it must learn reasoning patterns.

## What Makes Them "Large"?

The "large" in large language models refers to multiple dimensions of scale:

### Parameter Count

Modern LLMs contain billions or even trillions of parameters (weights) that are adjusted during training. For context:

- **GPT-2 (2019)**: 1.5 billion parameters
- **GPT-3 (2020)**: 175 billion parameters
- **GPT-4 (2023)**: Estimated trillions of parameters (exact figure not disclosed)
- **LLaMA 2 (2023)**: 7B, 13B, and 70B parameter variants

These parameters are organized in the **transformer architecture**, which we explored in an earlier topic. The attention mechanisms, feed-forward layers, and embedding tables all contribute to this massive parameter count.

### Training Data

The other dimension of scale is training data. GPT-3 was trained on approximately 300 billion tokens, while more recent models have consumed trillions of tokens. This data comes from diverse sources:

- Web pages (Common Crawl, curated subsets)
- Books (fiction and non-fiction)
- Wikipedia and reference materials
- Code repositories (GitHub)
- Scientific papers
- Social media and forums

### Compute Requirements

Training these models requires extraordinary computational resources. GPT-3's training reportedly cost millions of dollars in compute alone. This has led to a concentration of cutting-edge AI development among well-funded labs and large tech companies, raising important questions about democratization and access.

## The Emergence of Capabilities

Perhaps the most fascinating aspect of large language models is the phenomenon of **emergent capabilities**—abilities that appear suddenly as models reach certain scales, seemingly unpredictable from smaller models' behavior.

### Few-Shot Learning

One of the first emergent capabilities researchers noticed was few-shot learning. Give a sufficiently large language model a few examples of a task embedded in the prompt, and it can generalize to new examples without any fine-tuning:

```
Translate English to French:
English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: The weather is nice today.
French: Le temps est beau aujourd'hui.

English: I love learning about AI.
French:
```

The model completes this with "J'adore apprendre sur l'IA." without ever being explicitly trained on translation. It learned this pattern from its massive training corpus.

### Chain-of-Thought Reasoning

Another emergent capability is the ability to perform step-by-step reasoning when prompted appropriately. Rather than jumping to an answer, large models can work through problems:

```
Q: If John has 3 apples and gives half to Mary, then buys 4 more,
   how many apples does John have?

Let me work through this step by step:
1. John starts with 3 apples
2. He gives half to Mary: 3 / 2 = 1.5, so he gives 1 or 2 apples
   (assuming whole apples, let's say 1)
3. After giving away 1 apple, John has 2 apples
4. He buys 4 more: 2 + 4 = 6 apples
John has 6 apples.
```

This capability—where breaking down reasoning improves accuracy—wasn't present in smaller models and seems to emerge around the 100 billion parameter scale.

### Code Generation

Large language models trained on code (like Codex, which powers GitHub Copilot) can generate functional programs from natural language descriptions. This has transformed software development, with AI assistants now helping programmers worldwide write, debug, and understand code.

## The Training Process

Training an LLM involves several stages:

### Pre-training

The first stage is unsupervised pre-training on massive text corpora. The model learns to predict the next token, adjusting its billions of parameters to minimize prediction error. This stage requires the most compute and produces a "base model" that can complete text but isn't yet optimized for following instructions.

### Fine-tuning

Base models are then fine-tuned on more carefully curated data. This might include:

- **Instruction tuning**: Training on examples of instructions and appropriate responses
- **Reinforcement Learning from Human Feedback (RLHF)**: Using human preferences to guide model behavior
- **Constitutional AI**: Training models to follow explicit principles

These fine-tuning stages transform a text completion engine into a helpful assistant.

## The Transformer Architecture at Scale

LLMs are built on the transformer architecture (covered in a previous topic), but scaling up introduces specific engineering challenges:

### Attention Complexity

The self-attention mechanism has O(n²) complexity with respect to sequence length, meaning processing long documents becomes expensive. Various techniques address this:

- **Sparse attention patterns**: Attending to only a subset of tokens
- **Flash Attention**: Memory-efficient attention computation
- **Sliding window attention**: Attending to local context plus special global tokens

### Distributed Training

No single GPU can hold a 175 billion parameter model. Training requires distributing the model across hundreds or thousands of accelerators using techniques like:

- **Data parallelism**: Different GPUs process different batches
- **Model parallelism**: Different GPUs hold different parts of the model
- **Pipeline parallelism**: Different stages of computation on different devices

### Inference Optimization

Running inference on large models also requires significant resources. Techniques for making inference practical include:

- **Quantization**: Using lower-precision numbers (8-bit, 4-bit) instead of 32-bit floats
- **KV-cache**: Caching intermediate computations during generation
- **Speculative decoding**: Using smaller models to draft tokens verified by larger ones

## Why This Matters

Large language models represent a paradigm shift in AI. Rather than carefully engineering solutions for specific tasks, we now train general-purpose models that can be adapted to countless applications through prompting and fine-tuning.

This has democratized access to AI capabilities. Anyone who can write a prompt can now leverage sophisticated language understanding and generation. At the same time, it has raised profound questions:

- **Understanding vs. Mimicry**: Do these models truly understand language, or are they sophisticated pattern matchers?
- **Knowledge vs. Retrieval**: When an LLM "knows" something, what does that mean?
- **Creativity vs. Remixing**: Can statistical models produce genuinely novel content?

We'll explore some of these questions in the lesson on capabilities and limitations.

## Key Takeaways

1. **Large language models are neural networks trained to predict the next token** in a sequence, but this simple objective leads to remarkable capabilities when applied at scale.

2. **Scale matters across multiple dimensions**: parameter count, training data, and compute all contribute to model capabilities.

3. **Emergent capabilities appear at sufficient scale**, including few-shot learning, chain-of-thought reasoning, and code generation.

4. **The transformer architecture is the foundation** of modern LLMs, but scaling it up requires sophisticated engineering.

5. **LLMs represent a paradigm shift** from task-specific AI to general-purpose systems adapted through prompting.

## Further Reading

- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020) - Empirical laws governing LLM scaling
- "GPT-3: Language Models are Few-Shot Learners" (Brown et al., 2020) - The paper that sparked the LLM revolution
- "Emergent Abilities of Large Language Models" (Wei et al., 2022) - Analysis of capabilities that appear with scale
- "Attention Is All You Need" (Vaswani et al., 2017) - The original transformer paper
