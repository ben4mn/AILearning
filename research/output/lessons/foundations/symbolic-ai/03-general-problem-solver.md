# The General Problem Solver

## Introduction

The Logic Theorist could prove theorems, but only theorems. What about other problems—puzzles, planning, mathematical manipulations? Newell and Simon believed the same underlying mechanisms could tackle any well-defined problem.

In 1957, they began work on the **General Problem Solver (GPS)**—a program designed to exhibit general-purpose intelligence. Unlike specialized systems, GPS would embody problem-solving as a domain-independent process. Give it a problem description, and it would find a solution.

GPS never achieved its grand ambitions. But it introduced ideas—means-ends analysis, goal hierarchies, and the separation of problem content from solving methods—that remain central to AI and cognitive science.

## The Vision: General-Purpose Problem Solving

Newell and Simon observed that human problem-solving, across diverse domains, seemed to follow similar patterns:
- Identify the goal
- Assess the current state
- Find the difference between them
- Take actions to reduce that difference
- Repeat until the goal is reached

This was **means-ends analysis**: choose means (actions) based on their ability to reduce the difference between current state and desired ends (goals).

If this pattern was truly general, it could be programmed once and applied to any problem. That was GPS's promise.

## Means-Ends Analysis

The core of GPS was means-ends analysis (MEA). Here's how it worked:

### The Algorithm

```
1. GOAL: Achieve state G
2. Current state: S
3. DIFFERENCE: Compute D = difference(S, G)
4. If D is empty, success!
5. REDUCE: Find an operator O that reduces D
6. SUBGOAL: The operator may have preconditions P
7. If P not satisfied in S, create subgoal to achieve P
8. Apply O (if preconditions met)
9. Repeat from step 2 with new current state
```

### Example: The Monkey and Bananas Problem

A classic AI puzzle: A monkey is in a room with bananas hanging from the ceiling. There's a box the monkey can climb on. How does the monkey get the bananas?

```
Initial State:
- Monkey at location A
- Box at location B
- Bananas at location C (high)

Goal:
- Monkey has bananas

Difference:
- Monkey doesn't have bananas

Relevant Operator:
- Grasp(bananas) - requires monkey at C and monkey is high

Subgoal:
- Monkey at C and monkey is high

New Difference:
- Monkey not at C
- Monkey not high

...and so on, recursively creating subgoals
```

GPS would work backward and forward, creating a hierarchy of subgoals until it found a sequence of actions leading from initial state to goal.

### Operator Tables

GPS used **operator tables** that specified:
- The operator's name (e.g., Move, Push, Climb)
- Preconditions (what must be true to apply it)
- Effects (what changes when applied)
- Differences it can reduce

These tables constituted the problem-specific knowledge. The means-ends analysis algorithm was problem-independent.

```python
# Simplified GPS-style operator representation
class Operator:
    def __init__(self, name, preconditions, add_effects, del_effects, reduces):
        self.name = name
        self.preconditions = preconditions  # Required state
        self.add_effects = add_effects      # Facts added
        self.del_effects = del_effects      # Facts removed
        self.reduces = reduces              # Differences this operator can help with

# Example operator
move = Operator(
    name="Move(monkey, from, to)",
    preconditions=["At(monkey, from)"],
    add_effects=["At(monkey, to)"],
    del_effects=["At(monkey, from)"],
    reduces=["MonkeyNotAtLocation"]
)
```

## GPS in Practice

GPS was tested on several problem domains:

### Symbolic Integration

GPS could perform symbolic integration of mathematical expressions—transforming integrands step by step until reaching a solved form.

### Propositional Logic

Like the Logic Theorist, GPS could prove theorems, treating proof as search through the space of derivations.

### Tower of Hanoi

The classic puzzle of moving disks between pegs, one at a time, never placing a larger disk on a smaller one. GPS's means-ends analysis naturally handled the recursive structure.

### Missionaries and Cannibals

Three missionaries and three cannibals must cross a river. The boat holds two. If cannibals ever outnumber missionaries on either bank, the missionaries are eaten. GPS found safe crossing sequences.

## Achievements and Insights

GPS demonstrated several important ideas:

### Domain Independence

The same MEA algorithm worked across different domains. Only the operator tables changed. This suggested that problem-solving might be separable into domain-general methods and domain-specific knowledge.

### Goal Hierarchies

Complex problems decomposed into subgoals, sub-subgoals, and so on. This hierarchical structure seemed to match human problem-solving protocols—when people think aloud while solving problems, they mention goals and subgoals.

### Weak Methods

GPS embodied what Newell and Simon called "weak methods"—general techniques like MEA that work broadly but not deeply. Weak methods contrasted with "strong methods"—domain-specific expertise that works powerfully but narrowly.

### Computational Psychology

GPS was also a model of human cognition. Simon and Newell analyzed verbal protocols—recordings of people thinking aloud—and found striking parallels to GPS's behavior. This launched computational cognitive science.

## Limitations

Despite its elegance, GPS had serious limitations:

### The Problem of Problem Representation

GPS needed problems formalized in its language—states, operators, differences. But real-world problems don't come pre-formalized. Who decides what operators exist? What constitutes a "difference"? This representation problem proved thorny.

### Combinatorial Explosion

For complex problems, the search space exploded. GPS was brute-force compared to human expertise. A chess master doesn't search through all possible moves—they recognize patterns. GPS lacked this recognition ability.

### Brittleness

GPS couldn't handle novelty, ambiguity, or ill-defined problems. Real intelligence adapts to surprise; GPS failed ungracefully when problems didn't fit its expectations.

### Knowledge Acquisition

Where do operator tables come from? GPS needed them hand-coded by experts. Automatically acquiring this knowledge—the knowledge acquisition bottleneck—would plague AI for decades.

### No Learning

GPS didn't improve with experience. Each problem was approached fresh. Human intelligence accumulates expertise; GPS started from scratch every time.

## Historical Context

GPS was developed at RAND Corporation and Carnegie Tech in the late 1950s. It was programmed in IPL-II (an evolution of Shaw's IPL) and later in IPL-V.

The project continued through various versions until the mid-1960s. By then, the researchers had learned what they could from it and moved toward more powerful architectures.

GPS was never a practical system. It was a research vehicle—a way to explore ideas about intelligence. In this role, it succeeded brilliantly.

## Legacy

GPS's influence extends far beyond its immediate achievements:

### Planning in AI

Modern AI planning systems—used in robotics, logistics, game AI—descend from GPS. The STRIPS planner (1971) adopted GPS-style operator representations. Contemporary planners still use similar ideas.

### Production Systems

GPS led to production system architectures—rule-based systems that match conditions and fire actions. These evolved into expert systems in the 1970s-80s and cognitive architectures like SOAR and ACT-R.

### Cognitive Science

The idea of mind as information processor, manipulating symbol structures through methods like MEA, became a founding paradigm of cognitive science. Even critics who reject the paradigm engage with GPS's vision.

### Separation of Concerns

The distinction between domain-independent methods and domain-specific knowledge structured AI research. It encouraged work on both general algorithms and knowledge engineering.

### Hierarchical Task Decomposition

Breaking complex tasks into subtasks, recursively, is now standard in software engineering, project management, and robotics—all influenced by GPS's goal hierarchy ideas.

## From GPS to Modern AI

GPS was abandoned not because it failed but because researchers moved on. Its ideas were absorbed into the field:

**Planning**: STRIPS, UCPOP, GraphPlan, and modern planners refined GPS's approach.

**Expert Systems**: MYCIN, XCON, and other 1970s-80s systems applied specialized knowledge to specific domains.

**Cognitive Architectures**: SOAR, ACT-R, and others built more sophisticated models of cognition inspired by GPS.

**Modern AI**: While deep learning differs radically from GPS, questions about goals, subgoals, and hierarchical planning remain relevant.

GPS showed that general-purpose problem-solving was possible in principle. Whether it was the right path to intelligence remained—and remains—debated.

## Key Takeaways

- The General Problem Solver (1957-1969) was Newell and Simon's attempt at domain-independent problem-solving
- Its core method, means-ends analysis, reduced differences between current and goal states through operator application
- GPS created hierarchies of subgoals, decomposing complex problems into simpler ones
- It worked on various domains: symbolic math, logic, puzzles
- Limitations included representation problems, combinatorial explosion, brittleness, and no learning
- GPS's ideas—planning operators, goal hierarchies, weak methods—remain influential in AI and cognitive science

## Further Reading

- Newell, Allen & Simon, Herbert. *Human Problem Solving* (1972) - The comprehensive account of GPS and human cognition
- Ernst, G.W. & Newell, Allen. *GPS: A Case Study in Generality and Problem Solving* (1969) - Detailed technical description
- Russell, Stuart & Norvig, Peter. *Artificial Intelligence: A Modern Approach* (4th ed., 2020) - Chapter 11 on planning
- Simon, Herbert. *The Sciences of the Artificial* (3rd ed., 1996) - Broader perspective on artificial systems

---
*Estimated reading time: 8 minutes*
