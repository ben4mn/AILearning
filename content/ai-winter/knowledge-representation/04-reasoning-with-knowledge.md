# Reasoning with Knowledge

## Introduction

Having knowledge isn't enough—you need to use it. If you know that birds fly and that Tweety is a bird, you should be able to conclude that Tweety flies. This process of deriving new knowledge from existing knowledge is called inference or reasoning.

But reasoning in the real world is trickier than textbook logic suggests. What if Tweety is a penguin? What if you're not sure whether Tweety is a bird? What if the knowledge itself is inconsistent?

These questions drove decades of research into knowledge-based reasoning, producing insights that remain relevant even in the age of machine learning.

## Classical Deductive Reasoning

### Logic as Foundation

Classical AI reasoning was based on mathematical logic:

**Modus Ponens**: If P implies Q, and P is true, then Q is true
```
RULE: Human(X) → Mortal(X)
FACT: Human(Socrates)
CONCLUSION: Mortal(Socrates)
```

**Universal Instantiation**: What's true for all applies to each
```
∀X: Bird(X) → CanFly(X)
Bird(Tweety)
CONCLUSION: CanFly(Tweety)
```

### Forward and Backward Chaining

**Forward chaining** (data-driven): Start with facts, apply rules, derive conclusions
```
Facts: A, B
Rules: A ∧ B → C, C → D
Chain: A ∧ B triggers C; C triggers D
Result: We know A, B, C, D
```

**Backward chaining** (goal-driven): Start with a goal, find rules that conclude it, prove their premises
```
Goal: Prove D
Rule: C → D, so new goal: Prove C
Rule: A ∧ B → C, so new goals: Prove A, Prove B
Facts: A and B are known
Result: D is proved
```

### Limitations of Classical Logic

Classical logic was monotonic—adding information never withdraws conclusions:
- If you know Tweety flies, learning new facts can't make that false
- But in real reasoning, new information does change conclusions

It was also binary—things were true or false, never uncertain:
- But real knowledge is often "probably" or "usually"

## Non-Monotonic Reasoning

### The Problem

Real reasoning isn't monotonic. Consider:

**Before**: Tweety is a bird. Birds fly. Therefore Tweety flies.
**After**: Tweety is a penguin. Penguins don't fly. Therefore Tweety doesn't fly.

Adding "Tweety is a penguin" withdrew the conclusion "Tweety flies." Classical logic couldn't handle this.

### Default Logic

Raymond Reiter (1980) proposed default logic:

**Default rule**: "If X is a bird and it's consistent to assume X flies, conclude X flies"
```
Bird(X) : MCanFly(X)
─────────────────────
    Flies(X)

(Read: If X is a bird, and it's consistent to assume X can fly,
 then conclude X flies)
```

If you later learn Tweety is a penguin (and penguins can't fly), it's no longer consistent to assume Tweety can fly, so the conclusion is withdrawn.

### Circumscription

John McCarthy (1980) proposed circumscription:
- Assume abnormalities are minimized
- Things are normal unless known to be abnormal

```
Bird(X) ∧ ¬Abnormal(X) → Flies(X)
Penguin(X) → Abnormal(X)
```

Tweety flies unless there's evidence of abnormality (like being a penguin).

### Autoepistemic Logic

Robert Moore (1985) modeled reasoning about one's own beliefs:
- "If I don't believe Tweety is abnormal, I'll conclude Tweety flies"
- Conclusions depend on what you believe you know

### Practical Impact

Non-monotonic reasoning influenced:
- Expert system exception handling
- Database closed-world assumption
- AI planning with incomplete information

But the computational complexity was often prohibitive.

## Reasoning Under Uncertainty

### Why Uncertainty?

Real knowledge is uncertain:
- "Patients with these symptoms usually have the flu" (not always)
- "This stock will probably rise" (but might not)
- "The sensor reading is approximately 7.3" (with measurement error)

Binary logic couldn't capture this.

### Certainty Factors

MYCIN used certainty factors (see Expert Systems topic):
- Values from -1 (definitely false) to +1 (definitely true)
- Combined using special formulas

Simple but ad hoc—the formulas weren't principled.

### Bayesian Reasoning

Probabilistic approaches used Bayes' theorem:

```
P(Disease | Symptom) = P(Symptom | Disease) × P(Disease)
                       ─────────────────────────────────
                               P(Symptom)
```

**Advantages**:
- Principled mathematical foundation
- Clear meaning of uncertainty
- Consistent combination of evidence

**Challenges**:
- Requires probability estimates
- Computational complexity for large networks
- Experts often can't provide accurate probabilities

### Bayesian Networks

Judea Pearl's Bayesian networks (1980s) provided practical probabilistic reasoning:

```
         Flu?
        /    \
       ↓      ↓
   Fever?   Cough?
```

Each variable's probability depends only on its parents. This structure enabled efficient inference.

### Dempster-Shafer Theory

Dempster-Shafer theory (1970s) handled uncertainty differently:
- Assign belief masses to sets of possibilities
- Distinguish between "I believe it's true" and "I don't know"

Used in sensor fusion and some expert systems.

### Fuzzy Logic

Lotfi Zadeh's fuzzy logic (1965) handled vague concepts:
- Instead of "tall = true/false"
- "Degree of tallness" from 0 to 1

```
Tall(6'2") = 0.9
Tall(5'10") = 0.5
Tall(5'4") = 0.2
```

Widely used in control systems (fuzzy controllers) and some expert systems.

## Case-Based Reasoning

### A Different Approach

What if instead of rules, you reasoned from past cases?

**Human expert**: "This looks like the Smith case from last year. In that case, we did X and it worked."

Case-based reasoning (CBR) formalized this:
1. **Retrieve** similar past cases
2. **Reuse** the solution from the best-matching case
3. **Revise** the solution for the current situation
4. **Retain** the new case for future use

### Architecture

A CBR system contains:

**Case library**: Past problems and solutions
**Similarity metric**: How to compare cases
**Adaptation rules**: How to modify solutions

### Example: Help Desk

```
CASE-42:
  Problem: Printer won't print
  Symptoms: Paper jam light on
  Solution: Open rear panel, remove jammed paper
  Outcome: Resolved

NEW-PROBLEM:
  Symptoms: Paper jam light on
  → Retrieve CASE-42 (similar symptoms)
  → Reuse solution: Check rear panel
```

### Applications

CBR was used in:
- Help desks and technical support
- Legal reasoning (case law)
- Design (adapting past designs)
- Planning (reusing past plans)

### Strengths and Weaknesses

**Strengths**:
- Uses actual experience, not abstracted rules
- Handles novel situations through adaptation
- Naturally acquires knowledge through retention

**Weaknesses**:
- Needs good similarity metric
- Case library can grow unwieldy
- Adaptation can be complex

## Planning and Reasoning About Action

### Classical Planning

Planning meant reasoning about actions to achieve goals:

**Problem**:
- Initial state: I'm at home, package is at post office
- Goal: Package is at my home
- Actions: Go(X,Y), PickUp(X), PutDown(X)

**Solution**: Go(Home, PostOffice), PickUp(Package), Go(PostOffice, Home), PutDown(Package)

### STRIPS Representation

STRIPS (Stanford Research Institute Problem Solver, 1971) represented actions:

```
Action: Go(X, Y)
  Preconditions: At(X), Path(X, Y)
  Effects: At(Y), ¬At(X)

Action: PickUp(Obj)
  Preconditions: At(Location), At(Obj, Location), HandEmpty
  Effects: Holding(Obj), ¬At(Obj, Location), ¬HandEmpty
```

Planners searched for action sequences achieving goals.

### Challenges

**Frame problem**: Specifying what doesn't change with each action

**Ramifications**: Actions have indirect effects (moving a truck moves its cargo)

**Qualification problem**: Actions have prerequisites you might not know

**Interacting goals**: Multiple goals may conflict

### Plan Recognition

The inverse problem: given actions, infer the goal:

Observation: John got keys, went to car, drove to store
Inference: John's goal is probably to buy something at the store

Important for understanding stories and predicting behavior.

## Knowledge-Based Reasoning Today

### Hybrid Approaches

Modern systems combine:
- Logical inference for structured reasoning
- Statistical methods for uncertainty
- Neural networks for pattern recognition
- Knowledge graphs for background knowledge

### Question Answering

Systems like Watson combine:
- NLP to understand questions
- Retrieval from knowledge bases
- Inference to derive answers
- Confidence estimation

### Neuro-Symbolic AI

Current research integrates neural and symbolic:
- Neural networks that reason over knowledge graphs
- Symbolic constraints guiding neural learning
- Differentiable reasoning

The strict divide between statistical and symbolic approaches is blurring.

## Key Takeaways

- Deductive reasoning uses logic (modus ponens, universal instantiation) to derive conclusions
- Non-monotonic reasoning handles default assumptions that can be withdrawn: default logic, circumscription
- Uncertainty reasoning uses probabilistic methods: Bayesian networks, certainty factors, Dempster-Shafer
- Fuzzy logic handles vague concepts with degrees of truth
- Case-based reasoning solves new problems by adapting solutions from similar past cases
- Planning reasons about actions to achieve goals, facing the frame problem and qualification problem
- Modern AI increasingly combines multiple reasoning approaches in hybrid systems

## Further Reading

- Russell, Stuart & Norvig, Peter. *Artificial Intelligence: A Modern Approach* (4th ed., 2021) - Chapters on reasoning
- Pearl, Judea. *Probabilistic Reasoning in Intelligent Systems* (1988) - Bayesian networks
- Reiter, Raymond. "A Logic for Default Reasoning." *Artificial Intelligence* 13 (1980)
- Kolodner, Janet. *Case-Based Reasoning* (1993) - Comprehensive treatment

---
*Estimated reading time: 9 minutes*
