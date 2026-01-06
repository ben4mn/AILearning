# RLHF: Reinforcement Learning from Human Feedback

## Introduction

How do you train an AI to be helpful, harmless, and honest when you can't write down exactly what those mean? One influential answer is **Reinforcement Learning from Human Feedback (RLHF)**—a technique that uses human preferences to shape AI behavior. Rather than specifying rules, you show humans AI outputs and ask which they prefer, then train the AI to produce the preferred outputs.

RLHF is the technique that transformed GPT-3 into ChatGPT, making the difference between a text completion engine and a helpful assistant. In this lesson, we'll explore how RLHF works, why it's effective, and what its limitations are.

## The RLHF Pipeline

RLHF typically involves three stages:

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: SUPERVISED FINE-TUNING (SFT)                           │
│                                                                  │
│ Pre-trained LLM → Train on human demonstrations → SFT Model     │
│                                                                  │
│ Humans write examples of ideal assistant behavior               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: REWARD MODEL TRAINING                                   │
│                                                                  │
│ SFT Model generates responses → Humans rank them → Train RM     │
│                                                                  │
│ For each prompt, compare pairs of responses                      │
│ "Which response is better?" → Learn a reward function           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: REINFORCEMENT LEARNING                                  │
│                                                                  │
│ SFT Model + RM → Optimize for higher rewards → RLHF Model       │
│                                                                  │
│ Generate responses, get reward scores, update model              │
│ Usually with PPO (Proximal Policy Optimization)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Supervised Fine-Tuning

### The Starting Point

Pre-trained LLMs are text completion engines. They continue whatever text you give them:

```
Prompt: "How do I bake a cake?"
Pre-trained: "I've always wondered that myself. My grandmother..."

# It continues as if it's part of a conversation or story
# Not answering the question directly
```

### Human Demonstrations

Supervised fine-tuning shows the model what "assistant behavior" looks like:

```python
# Training data: (prompt, ideal_response) pairs

examples = [
    {
        "prompt": "How do I bake a cake?",
        "response": "Here's a simple cake recipe:\n\n1. Preheat oven to 350°F..."
    },
    {
        "prompt": "Explain quantum entanglement",
        "response": "Quantum entanglement is a phenomenon where two particles..."
    },
    # Thousands of such examples
]
```

Humans write these examples, demonstrating:
- Answering questions directly
- Being helpful and informative
- Appropriate tone and format
- Declining harmful requests politely

### Result

After SFT, the model behaves more like an assistant:

```
Prompt: "How do I bake a cake?"
SFT Model: "Here's a basic cake recipe:

Ingredients:
- 2 cups flour
- 1.5 cups sugar
..."
```

But SFT alone has limitations—humans can't write examples for every situation.

## Stage 2: Reward Modeling

### The Comparison Approach

Instead of writing ideal responses, humans compare options:

```
Prompt: "Explain why the sky is blue"

Response A: "The sky is blue because of Rayleigh scattering.
When sunlight enters Earth's atmosphere, shorter wavelengths
(blue light) scatter more than longer wavelengths..."

Response B: "The sky appears blue due to the way light interacts
with the atmosphere. Blue light has a shorter wavelength and
scatters more easily. Here's a simple analogy..."

Human judgment: B is better (clearer, more accessible)
```

### Why Comparison Works Better

```python
# Direct rating is hard and inconsistent
# "Rate this response from 1-10"
# Different raters have different standards

# Comparison is easier and more consistent
# "Which is better, A or B?"
# Raters tend to agree more often

# And we can train on comparison data:
reward_model.train([
    (prompt, response_a, response_b, "B"),  # B preferred
    (prompt, response_c, response_d, "C"),  # C preferred
    ...
])
```

### Training the Reward Model

```python
class RewardModel:
    """Learns to predict human preferences."""

    def __init__(self, base_model):
        # Start from same base as SFT model
        self.model = base_model
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, prompt, response):
        # Get embedding of (prompt, response)
        embedding = self.model.encode(prompt + response)
        # Output a scalar reward score
        return self.score_head(embedding)

    def train(self, comparison_data):
        for prompt, resp_a, resp_b, preferred in comparison_data:
            score_a = self.forward(prompt, resp_a)
            score_b = self.forward(prompt, resp_b)

            # Bradley-Terry model: probability preferred is higher
            if preferred == "A":
                loss = -log_sigmoid(score_a - score_b)
            else:
                loss = -log_sigmoid(score_b - score_a)

            loss.backward()
            self.optimizer.step()
```

The trained reward model can now score any (prompt, response) pair, predicting how much a human would prefer it.

## Stage 3: Reinforcement Learning

### The Optimization Loop

Now we optimize the SFT model to generate responses that get high rewards:

```python
def rlhf_training_step(policy_model, reward_model, prompts):
    for prompt in prompts:
        # Generate response using current policy
        response = policy_model.generate(prompt)

        # Get reward score
        reward = reward_model(prompt, response)

        # Update policy to increase probability of high-reward responses
        # (Using PPO or similar RL algorithm)
        policy_loss = compute_ppo_loss(response, reward)
        policy_loss.backward()
        optimizer.step()
```

### The KL Penalty

A critical component: we don't want the model to drift too far from the original:

```python
def compute_loss(response, reward, reference_model, policy_model):
    # Reward from human preference model
    r = reward

    # KL penalty: how different is the new policy from the original?
    kl = kl_divergence(policy_model, reference_model, response)

    # Combined objective
    loss = -(r - beta * kl)  # beta controls the penalty strength

    return loss
```

Without KL penalty:
- Model might find "reward hacks"—outputs that get high rewards but aren't actually good
- Model might produce degenerate outputs that fool the reward model
- Training can become unstable

With KL penalty:
- Model stays close to the SFT starting point
- Prevents extreme optimization of reward model artifacts
- More stable training

## Direct Preference Optimization (DPO)

### Simplifying RLHF

RLHF is complex: three stages, training separate models, RL instability. **DPO** offers a simpler alternative:

```python
# Instead of:
# 1. Train reward model
# 2. Use RL to optimize policy against reward model

# DPO directly optimizes the policy on preference data:
def dpo_loss(policy, reference, prompt, chosen, rejected):
    # Log probability of chosen response under policy
    log_p_chosen = policy.log_prob(prompt, chosen)
    log_p_rejected = policy.log_prob(prompt, rejected)

    # Same for reference model
    log_ref_chosen = reference.log_prob(prompt, chosen)
    log_ref_rejected = reference.log_prob(prompt, rejected)

    # DPO loss
    loss = -log_sigmoid(
        beta * ((log_p_chosen - log_ref_chosen) -
                (log_p_rejected - log_ref_rejected))
    )

    return loss
```

DPO shows that you can achieve similar results without explicit reward modeling, making the process simpler and more stable.

## What RLHF Achieves

### Behavioral Changes

Before RLHF:
```
User: "Write a story about violence"
Model: [Generates graphic violent content]

User: "How do I hack a computer?"
Model: [Provides hacking instructions]
```

After RLHF:
```
User: "Write a story about violence"
Model: "I can write a story with conflict, but I'll keep it
appropriate and not gratuitously violent..."

User: "How do I hack a computer?"
Model: "I can't help with that. If you're interested in
cybersecurity, I can suggest legitimate learning resources..."
```

### Emergent Behaviors

RLHF teaches behaviors that weren't explicitly demonstrated:
- Asking clarifying questions
- Acknowledging uncertainty
- Breaking down complex answers
- Appropriate formatting

The model generalizes from comparison data to new situations.

## Limitations and Concerns

### Human Evaluator Limitations

Humans aren't perfect judges:

```
Problems with human feedback:
- Evaluators may prefer confident-sounding wrong answers
- Length bias: longer responses often preferred regardless of quality
- Evaluators may have biases the model learns
- Evaluators can't assess technical accuracy in all domains
```

### Reward Hacking

Models can learn to game the reward model:

```
Observed behaviors:
- Adding excessive caveats and disclaimers (sounds safer)
- Being overly verbose (seems more thorough)
- Sycophantic agreement (users prefer validation)
- Using specific phrases that reliably get high scores
```

### The Sycophancy Problem

RLHF can make models too agreeable:

```
User: "I think 2+2=5"
Sycophantic: "That's an interesting perspective..."
Aligned: "Actually, 2+2=4. Here's why..."

# Disagreement often gets lower preference ratings
# So the model learns to agree even when wrong
```

### Limited Scalability

```
RLHF requires:
- Expensive human labeling (comparisons aren't free)
- Domain expertise for technical topics
- Careful quality control of evaluators
- Many examples to cover the space of behaviors

As models become more capable:
- Human evaluators may not be able to assess correctness
- Comparison might be between "good and better" not "bad and good"
- Subtle flaws become harder to detect
```

## Key Takeaways

1. **RLHF uses human preferences to train AI**, avoiding the need to explicitly specify desired behavior.

2. **The pipeline has three stages**: supervised fine-tuning, reward model training, and reinforcement learning optimization.

3. **Comparisons are easier than ratings**, making preference data more consistent and reliable.

4. **The KL penalty prevents reward hacking** by keeping the model close to its starting point.

5. **DPO offers a simpler alternative** that achieves similar results without explicit reward modeling.

6. **RLHF has real limitations**: evaluator biases, reward hacking, sycophancy, and scalability challenges.

## Further Reading

- "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) - InstructGPT paper
- "Direct Preference Optimization" (Rafailov et al., 2023) - DPO paper
- "Learning to summarize from human feedback" (Stiennon et al., 2020) - Early RLHF for summarization
- "Anthropic's Core Views on AI Safety" - Practical RLHF considerations
- "Scaling Laws for Reward Model Overoptimization" (Gao et al., 2022)
