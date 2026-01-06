# The Logic Theorist

## Introduction

On December 15, 1955, a program running on a Johnniac computer at RAND Corporation accomplished something unprecedented: it proved a theorem in mathematical logic. The theorem—that the sum of two even numbers is even—was trivial for humans. But a machine had never done anything like it.

The program was the **Logic Theorist** (LT), created by Allen Newell, Herbert Simon, and Cliff Shaw. It would go on to prove 38 of the first 52 theorems in Russell and Whitehead's *Principia Mathematica*, finding proofs more elegant than the originals for several.

The Logic Theorist was arguably the first artificial intelligence program. It demonstrated that machines could perform tasks that, in humans, required insight and creativity. The age of AI had begun.

## The Creators

Three men brought the Logic Theorist to life:

### Herbert Simon (1916-2001)

Simon was a polymath—economist, political scientist, psychologist, and computer scientist. He would win the Nobel Prize in Economics (1978), the Turing Award (1975), and numerous other honors.

Simon's core interest was decision-making: how do humans and organizations make choices? He developed the concept of "bounded rationality"—the idea that real decision-makers have limited information and cognitive resources, so they "satisfice" (find good-enough solutions) rather than optimize.

This perspective shaped his approach to AI: intelligence wasn't about finding perfect solutions but about clever search through possibilities.

### Allen Newell (1927-1992)

Newell was an operations researcher and computer scientist. At RAND Corporation, he had worked on air defense systems and became fascinated by the potential of computers for cognitive simulation.

Newell was the architectural thinker of the team. He designed the conceptual frameworks and worked out how ideas could be implemented. His later work on cognitive architectures (SOAR) continued this focus.

### Cliff Shaw (1922-1991)

Shaw was a RAND programmer who developed the list-processing language (IPL) that made the Logic Theorist possible. He solved crucial implementation problems and turned Newell and Simon's ideas into running code.

Shaw's role is often underappreciated. Without his programming innovations, the Logic Theorist would have remained theoretical.

## The Challenge: Proving Theorems

Russell and Whitehead's *Principia Mathematica* derived mathematical truths from a small set of axioms using precise inference rules. Could a computer do the same?

The problem seemed amenable to computation:
- Axioms are formal statements
- Inference rules are mechanical operations
- Theorems are derivable through rule application

But there was a catch: the space of possible derivations was enormous. Most paths led nowhere. A brute-force search would take longer than the universe's lifetime.

This is where Newell and Simon's insight mattered: intelligent problem-solving isn't exhaustive search. It's selective search guided by heuristics.

## Heuristic Search

The Logic Theorist introduced **heuristic search**—using rules of thumb to guide exploration toward promising directions.

### The Basic Strategy

LT worked backward from the goal (the theorem to prove) toward known truths (axioms and previously proven theorems):

1. Take the theorem to be proven
2. Find substitution rules or inference rules that could derive it
3. For each possibility, determine what sub-goals must be proven
4. Recursively prove sub-goals
5. If a chain reaches axioms or known theorems, success!

This backward chaining focused effort on relevant inference chains rather than wandering aimlessly from axioms.

### Heuristics Used

LT employed several heuristics:

**Similarity heuristic**: Prefer applying rules that make the current expression more similar to known truths. If the goal looks almost like a known theorem, try to make it identical.

**Detachment heuristic**: If you're trying to prove B, and you know A → B, try to prove A. This exploits the structure of implications.

**Substitution heuristic**: If the goal and a known theorem differ only in variable names, the goal might be provable by substitution.

**Chaining heuristic**: If A → B is known and you want C → B, try to prove C → A.

These heuristics didn't guarantee finding a proof, but they dramatically reduced the search space.

```
Example of backward reasoning:

Goal: Prove (p ∨ q) → (q ∨ p)

Known theorem: (a ∨ b) → (b ∨ a) [with generic variables]

Similarity heuristic notices: Goal matches known theorem!

Try substitution: a = p, b = q

Result: Known theorem directly proves goal.
```

## Technical Implementation

The Logic Theorist ran on the Johnniac computer (named after von Neumann) at RAND. Shaw developed **IPL (Information Processing Language)** to support the program.

### IPL Innovations

IPL introduced concepts now standard in programming:
- **List structures**: Dynamic, linked data structures
- **Associative retrieval**: Finding items by properties, not just location
- **Recursion**: Functions that call themselves
- **Symbol manipulation**: Processing symbolic expressions, not just numbers

These features were essential for AI programming and influenced later languages, especially LISP.

### Memory and Processing

The Johnniac had limited memory by modern standards—about 36,000 words. LT's search had to be carefully managed to fit. Shaw's clever data structures and garbage collection were crucial.

Execution was slow—proving a theorem could take minutes to hours. But it worked.

## Landmark Results

The Logic Theorist's achievements were remarkable:

**38 of 52 theorems proven**: LT proved 38 of the first 52 theorems in Chapter 2 of *Principia Mathematica*. The 14 it failed on were generally more complex or required techniques beyond its repertoire.

**Novel proofs discovered**: For some theorems, LT found proofs shorter or more elegant than Russell and Whitehead's originals. This was creative—the machine had discovered something its creators hadn't anticipated.

**Theorem 2.85**: LT's proof of this theorem was so elegant that Simon reportedly submitted it to the *Journal of Symbolic Logic*. The journal rejected it—apparently not ready for machine-authored mathematics.

## Reception and Impact

The Logic Theorist generated excitement but also skepticism:

### At Dartmouth

Newell and Simon presented LT at the 1956 Dartmouth Conference. It was the most concrete demonstration of machine intelligence. While some attendees were impressed, others were cooler—perhaps skeptical that theorem proving counted as "real" intelligence.

### Simon's Claims

Simon was characteristically bold:

> "We have invented a computer program capable of thinking non-numerically, and thereby solved the venerable mind-body problem."

This claim—that they had solved the mind-body problem—was grandiose but captured the excitement. If a machine could prove theorems creatively, wasn't it thinking?

### The AI Research Agenda

LT established a template for early AI:
1. Choose a cognitive task (proving, playing, planning)
2. Represent knowledge symbolically
3. Use heuristic search to find solutions
4. Evaluate on specific benchmarks

This approach dominated AI for three decades.

## Lessons from the Logic Theorist

Several insights emerged from LT:

### Heuristics are Essential

Brute force fails. Intelligence requires selectivity—focusing computational resources on promising paths. This insight underpins all of AI.

### Representation Matters

How you represent a problem affects how hard it is to solve. LT's symbolic representation of logical statements enabled its inference methods.

### Creativity is Possible

LT found novel proofs, demonstrating that machines could surprise their creators. This challenged the view that machines could only do what they were explicitly programmed to do.

### Understanding Through Building

Newell and Simon learned about human problem-solving by building a problem-solver. This methodology—computational models of cognition—became a research paradigm.

## Legacy

The Logic Theorist's influence extends far beyond theorem proving:

**Cognitive Science**: Simon and Newell founded cognitive science as the study of mind through computational models. Their book *Human Problem Solving* (1972) analyzed human thinking using AI concepts.

**Search Algorithms**: LT's heuristic search ideas evolved into A* search, branch-and-bound, and other fundamental algorithms.

**Expert Systems**: Later knowledge-based systems applied similar ideas to practical domains—medical diagnosis, configuration, planning.

**AI Methodology**: The practice of building running systems and evaluating them empirically became AI's standard methodology.

The Logic Theorist didn't solve AI—the gap between proving simple theorems and general intelligence remained vast. But it proved that computational intelligence was possible and pointed toward methods that would be developed for decades.

## Key Takeaways

- The Logic Theorist (1955-1956), by Newell, Simon, and Shaw, was arguably the first AI program
- It proved theorems from Russell and Whitehead's *Principia Mathematica* using heuristic search
- Key innovation: selective search guided by heuristics, not brute-force enumeration
- IPL, the implementation language, introduced list structures and symbolic processing
- LT found novel proofs, demonstrating machine creativity
- The methodology—symbolic representation plus heuristic search—defined early AI

## Further Reading

- Newell, Allen & Simon, Herbert. "The Logic Theory Machine: A Complex Information Processing System." *IRE Transactions on Information Theory* 2, no. 3 (1956): 61-79
- McCorduck, Pamela. *Machines Who Think* (2nd ed., 2004) - Extensive interviews with Simon and Newell
- Simon, Herbert. *Models of My Life* (1991) - Autobiography covering the LT era
- Crevier, Daniel. *AI: The Tumultuous History of the Search for Artificial Intelligence* (1993) - Chapter on early programs

---
*Estimated reading time: 8 minutes*
