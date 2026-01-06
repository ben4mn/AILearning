# Logic and Reasoning

## Introduction

At the heart of early AI was a seductive idea: thinking is a form of logic. If we could formalize the rules of valid reasoning, we could program machines to reason. This wasn't just optimism—it had a centuries-long pedigree.

From Aristotle's syllogisms to Boole's algebra to Russell and Whitehead's monumental *Principia Mathematica*, logicians had built ever more powerful systems for capturing valid inference. Now, with digital computers, these systems could come alive. Machines could manipulate symbols according to logical rules and, perhaps, in doing so, think.

In this lesson, we'll explore the logical foundations that made symbolic AI possible—the formal systems that researchers believed could capture the essence of intelligence.

## Logic Before Computers

### Aristotle's Syllogisms

Western logic began with Aristotle (384-322 BCE), who systematized valid forms of argument. His syllogisms captured patterns like:

```
All men are mortal.        (Major premise)
Socrates is a man.         (Minor premise)
Therefore, Socrates is mortal.  (Conclusion)
```

This is valid regardless of what you substitute for "men," "mortal," and "Socrates." The form guarantees the conclusion follows from the premises.

Aristotle catalogued dozens of valid forms. For two millennia, this was the core of logic education. But syllogistic logic was limited—it couldn't express many mathematical and everyday inferences.

### Boolean Algebra

In 1847, George Boole published *The Mathematical Analysis of Logic*, showing that logical reasoning could be reduced to algebra. Propositions became variables (0 for false, 1 for true). Logical operations became mathematical functions:

- AND: A ∧ B = A × B
- OR: A ∨ B = A + B - A × B
- NOT: ¬A = 1 - A

This was revolutionary: logic became mathematics. Calculations could replace intuition. Machines could, in principle, perform logical operations through electrical circuits—as Claude Shannon would later demonstrate.

### Predicate Logic

Propositional logic handles simple statements, but can't express relations or quantities. Consider: "Every student in the class passed the exam." This can't be captured propositionally.

Gottlob Frege and later Bertrand Russell developed **predicate logic** (or first-order logic), which added:

- **Variables**: x, y, z representing objects
- **Predicates**: Properties like Student(x), Passed(x, Exam)
- **Quantifiers**: ∀ (for all) and ∃ (there exists)

Now we can write:
```
∀x (Student(x) → Passed(x, Exam))
```

"For all x, if x is a student, then x passed the exam."

Predicate logic was far more expressive than propositional logic. It could represent much of mathematical reasoning and everyday inference.

### Principia Mathematica

Russell and Alfred North Whitehead spent a decade writing *Principia Mathematica* (1910-1913), attempting to derive all of mathematics from pure logic. The project was foundational: if mathematics was logic, and computers could do logic, then computers could do mathematics.

The work introduced formal proof systems—precise rules for deriving theorems from axioms. Each step was mechanical, checkable, repeatable. This was exactly what computers needed.

```
A formal proof system:
- Axioms: Starting truths
- Inference rules: Ways to derive new statements
- Theorems: Statements derived through rule application

Example (Modus Ponens):
  If A → B is proven
  And A is proven
  Then B can be derived
```

## Why Logic Seemed Like the Path to AI

When AI pioneers looked for a foundation, logic was attractive for several reasons:

### Universality

Logic claimed to capture all valid reasoning. If a conclusion follows from premises, logic can demonstrate it. This suggested intelligence might reduce to sophisticated logical inference.

### Precision

Logic was rigorous, formal, unambiguous. Programs could implement logical rules exactly. There was no handwaving about "somehow" reaching conclusions.

### Compositionality

Complex logical statements built from simple parts. The meaning of "∀x (Student(x) → Passed(x, Exam))" derives from its components. This modularity mapped well to programming.

### Existing Tools

Logicians had developed powerful proof methods over decades. These weren't just theoretical—they were procedures that, with effort, could be automated.

### The Hilbert Program

David Hilbert had proposed (before Gödel's incompleteness results) that all of mathematics could be formalized and proven consistent. Though Gödel showed this was impossible in its full ambition, the spirit of formalization remained powerful.

## Representing Knowledge Logically

Symbolic AI used logic to represent both knowledge and inference:

### Facts as Propositions

```
Father(John, Mary)     -- John is Mary's father
Human(Mary)            -- Mary is human
Age(Mary, 25)          -- Mary is 25 years old
```

### Rules as Implications

```
∀x∀y (Father(x, y) → Parent(x, y))
-- For all x, y: if x is father of y, then x is parent of y

∀x (Human(x) → Mortal(x))
-- All humans are mortal
```

### Goals as Queries

"Is Mary mortal?" becomes a query to derive: Mortal(Mary)

The system chains backward or forward through rules:
1. Human(Mary) is given
2. Human(x) → Mortal(x) is a rule
3. Substituting Mary for x: Human(Mary) → Mortal(Mary)
4. By modus ponens: Mortal(Mary)

```python
# A simple propositional reasoning example
knowledge_base = {
    "facts": ["Human(Mary)", "Father(John, Mary)"],
    "rules": [
        ("Human(x)", "Mortal(x)"),
        ("Father(x, y)", "Parent(x, y)")
    ]
}

def query(goal, kb):
    """
    Very simplified forward chaining.
    Real systems are more sophisticated.
    """
    derived = set(kb["facts"])

    changed = True
    while changed:
        changed = False
        for (condition, conclusion) in kb["rules"]:
            # In a real system, we'd handle variable binding
            # This is just illustrative
            for fact in list(derived):
                if matches(fact, condition):
                    new_fact = substitute(conclusion, fact)
                    if new_fact not in derived:
                        derived.add(new_fact)
                        changed = True

    return goal in derived
```

## Inference Methods

Several automated reasoning methods emerged:

### Resolution

J. Alan Robinson's 1965 resolution principle provided a single, powerful inference rule sufficient for first-order logic theorem proving. If you could refute the negation of what you wanted to prove, you'd proven it.

Resolution theorem provers became practical tools, though they struggled with computational complexity.

### Forward Chaining

Start with known facts. Apply rules to derive new facts. Repeat until the goal is derived or no new facts emerge.

This is data-driven: the system reacts to what it knows.

### Backward Chaining

Start with the goal. Find rules that could derive it. Recursively try to prove those rules' conditions.

This is goal-driven: the system focuses on what it needs to prove.

### Unification

When matching patterns, variables must be bound consistently. Unification finds the most general substitution that makes two expressions identical.

```
Unify: Parent(x, Mary) with Parent(John, y)
Result: x = John, y = Mary
Unified form: Parent(John, Mary)
```

## Limitations of Pure Logic

Even in AI's early days, limitations were recognized:

### The Frame Problem

If you move a block, what else changes? Logic represents facts, but doesn't automatically propagate changes. Specifying what stays the same requires exponentially many statements.

### Common Sense

Logic formalizes explicit knowledge. But humans rely on vast, implicit common sense: objects fall down, people have two arms, pouring water fills containers. Encoding this proved nearly impossible.

### Uncertainty

Classical logic is binary: true or false. Real reasoning involves probability, defaults, and revisions. Extensions like fuzzy logic and probabilistic reasoning were developed, but added complexity.

### Computational Complexity

Logical inference is often NP-hard or worse. Even with sophisticated heuristics, scaling to real-world knowledge bases proved difficult.

### Brittleness

Logic-based systems failed ungracefully. A single missing fact or incorrect rule could cause complete failure, with no way to approximate or guess.

## Logic's Legacy

Despite limitations, logic fundamentally shaped AI:

**Knowledge representation**: Modern AI systems still use structured representations inspired by logical formalisms.

**Planning**: Automated planning uses logical representations of actions, preconditions, and effects.

**Verification**: Formal methods in software engineering apply logical techniques to prove program correctness.

**Ontologies**: The Semantic Web and knowledge graphs descend from logical knowledge representation.

**Neuro-symbolic AI**: Current research explores combining neural learning with logical reasoning—the best of both paradigms.

Logic wasn't the complete answer to AI, but it provided concepts, tools, and a precise vocabulary that remain essential.

## Key Takeaways

- Formal logic developed over centuries, from Aristotle through Boole to Russell and Whitehead
- Predicate logic's expressiveness and rigor made it attractive as a foundation for AI
- Symbolic AI represented knowledge as logical facts and rules, with inference as theorem proving
- Key inference methods included resolution, forward chaining, backward chaining, and unification
- Limitations (frame problem, common sense, uncertainty, complexity, brittleness) prevented logic from being AI's complete solution
- Logic's concepts and tools remain influential in modern AI, including current neuro-symbolic approaches

## Further Reading

- Russell, Stuart & Norvig, Peter. *Artificial Intelligence: A Modern Approach* (4th ed., 2020) - Chapters 7-9 on logical agents
- McCarthy, John & Hayes, Patrick. "Some Philosophical Problems from the Standpoint of Artificial Intelligence." *Machine Intelligence* 4 (1969): 463-502
- Nilsson, Nils. *Artificial Intelligence: A New Synthesis* (1998) - Thorough coverage of logical AI
- Whitehead, A.N. & Russell, Bertrand. *Principia Mathematica* (1910-1913) - The classic work (warning: not light reading!)

---
*Estimated reading time: 9 minutes*
