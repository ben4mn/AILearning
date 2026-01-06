# Prolog and Logic Programming

## Introduction

While American AI adopted LISP, a different approach emerged in Europe. What if instead of telling computers *how* to solve problems, you could simply describe *what* the problem was? What if programs were logical statements, and running a program meant proving a theorem?

This was the vision of logic programming, and its most successful embodiment was Prolog—Programming in Logic. Created in 1972, Prolog offered a radically different way to think about computation, one that would influence AI, databases, and programming language theory.

## The Logic Programming Idea

### Declarative vs. Procedural

Traditional programming was *procedural*—you specified step-by-step instructions:

```python
# Procedural: How to find ancestors
def ancestors(person):
    result = []
    for parent in get_parents(person):
        result.append(parent)
        result.extend(ancestors(parent))
    return result
```

Logic programming was *declarative*—you specified facts and relationships:

```prolog
% Declarative: What ancestors are
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

The system figured out *how* to answer queries.

### Programs as Theories

In logic programming:
- A program is a set of logical statements (axioms)
- Running the program means answering a query
- Answering a query means proving it from the axioms

This aligned beautifully with symbolic AI's roots in mathematical logic.

## Origins of Prolog

### The Edinburgh-Marseille Collaboration

Prolog emerged from collaboration between:

**Robert Kowalski** (University of Edinburgh): Developed the theoretical foundations of logic programming, showing how Horn clauses could serve as a programming language.

**Alain Colmerauer** (University of Aix-Marseille): Built the first Prolog implementation in 1972, initially for natural language processing.

### The Breakthrough Insight

Kowalski's key insight was that a logical implication like:

```
C if A and B
```

Could be read two ways:

**Logical reading**: "C is true if A and B are true"

**Procedural reading**: "To prove C, first prove A, then prove B"

This dual reading made logic executable.

### Early Applications

Prolog was initially used for:
- Natural language parsing (Colmerauer's original interest)
- Theorem proving
- Expert systems
- Database querying

## Prolog Syntax and Semantics

### Facts

Facts state things that are simply true:

```prolog
% Facts about family relationships
parent(tom, mary).
parent(tom, john).
parent(mary, ann).
parent(mary, pat).
parent(john, jim).

% Facts about properties
male(tom).
male(john).
male(jim).
female(mary).
female(ann).
female(pat).
```

### Rules

Rules define relationships in terms of other relationships:

```prolog
% X is a father of Y if X is parent of Y and X is male
father(X, Y) :- parent(X, Y), male(X).

% X is a mother of Y if X is parent of Y and X is female
mother(X, Y) :- parent(X, Y), female(X).

% X is a grandparent of Y if X is parent of Z and Z is parent of Y
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).

% X is a sibling of Y if they share a parent and aren't the same
sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.

% Ancestor is defined recursively
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

### Queries

Queries ask the system to find solutions:

```prolog
?- parent(tom, mary).
yes.

?- parent(tom, Who).
Who = mary ;
Who = john.

?- grandparent(tom, Grandchild).
Grandchild = ann ;
Grandchild = pat ;
Grandchild = jim.

?- ancestor(tom, X).
X = mary ;
X = john ;
X = ann ;
X = pat ;
X = jim.
```

### Unification

Prolog's core operation was *unification*—finding variable assignments that make two terms identical:

```prolog
% Does foo(X, bar) unify with foo(baz, Y)?
% Yes: X = baz, Y = bar

% Does parent(tom, X) unify with parent(Y, mary)?
% Yes: Y = tom, X = mary

% Does foo(X, X) unify with foo(1, 2)?
% No: X can't be both 1 and 2
```

Unification was pattern matching on steroids.

### Backtracking

When Prolog tried to satisfy a goal and failed, it *backtracked*—undid choices and tried alternatives:

```prolog
?- parent(X, Y).
% Try first clause: X = tom, Y = mary
X = tom, Y = mary ;
% Backtrack, try second clause: X = tom, Y = john
X = tom, Y = john ;
% Continue through all possibilities...
```

This automatic search was powerful but could be inefficient.

## Prolog for AI

### Natural Language Processing

Prolog excelled at parsing:

```prolog
% Grammar rules
sentence(S) :- noun_phrase(NP), verb_phrase(VP),
               append(NP, VP, S).

noun_phrase(NP) :- determiner(D), noun(N),
                   append(D, N, NP).

verb_phrase(VP) :- verb(V), noun_phrase(NP),
                   append(V, NP, VP).

% Vocabulary
determiner([the]).
determiner([a]).
noun([cat]).
noun([dog]).
verb([chased]).
verb([saw]).

?- sentence([the, cat, chased, a, dog]).
yes.
```

Prolog's built-in search handled the combinatorics of parsing.

### Expert Systems

Prolog was natural for rule-based systems:

```prolog
% Medical diagnosis rules
diagnosis(flu) :-
    symptom(fever),
    symptom(body_aches),
    symptom(fatigue).

diagnosis(cold) :-
    symptom(runny_nose),
    symptom(sneezing),
    \+ symptom(fever).

recommend(rest) :- diagnosis(flu).
recommend(fluids) :- diagnosis(flu).
recommend(decongestant) :- diagnosis(cold).

% Interactive session
?- assert(symptom(fever)),
   assert(symptom(body_aches)),
   assert(symptom(fatigue)).
yes.

?- diagnosis(D).
D = flu.

?- recommend(R).
R = rest ;
R = fluids.
```

### Constraint Satisfaction

Prolog handled constraint problems elegantly:

```prolog
% N-Queens problem
n_queens(N, Qs) :-
    length(Qs, N),
    domain(Qs, 1, N),
    safe(Qs),
    labeling(Qs).

safe([]).
safe([Q|Qs]) :- no_attack(Q, Qs, 1), safe(Qs).

no_attack(_, [], _).
no_attack(Q, [Q1|Qs], D) :-
    Q =\= Q1,
    Q - Q1 =\= D,
    Q1 - Q =\= D,
    D1 is D + 1,
    no_attack(Q, Qs, D1).
```

### Database Querying

Prolog resembled relational databases:

```prolog
% Facts as database tables
employee(john, engineering, 50000).
employee(mary, sales, 60000).
employee(bob, engineering, 55000).

% Queries as Prolog goals
?- employee(Name, engineering, Salary).
Name = john, Salary = 50000 ;
Name = bob, Salary = 55000.

?- employee(Name, Dept, Salary), Salary > 52000.
Name = mary, Dept = sales, Salary = 60000 ;
Name = bob, Dept = engineering, Salary = 55000.
```

SQL was influenced by this logical approach to data.

## The Fifth Generation Project

### Japan's Bet on Prolog

In 1982, Japan launched the Fifth Generation Computer Systems (FGCS) project, a ten-year, $850 million effort to build advanced AI computers.

The project chose Prolog (in a dialect called Kernel Language) as its foundation:
- Logic programming for knowledge representation
- Parallel execution for performance
- Integration with knowledge bases

### International Impact

The announcement shocked the West:
- Was Japan about to leapfrog American AI?
- Should governments respond?
- Was Prolog the future?

The US created MCC (Microelectronics and Computer Technology Corporation); the UK launched the Alvey Programme; Europe started ESPRIT.

### The Outcome

The Fifth Generation project ultimately disappointed:
- Parallel Prolog proved difficult
- Knowledge systems didn't scale as hoped
- The AI winter of the late 1980s dimmed enthusiasm

But Prolog itself survived and evolved.

## Prolog vs. LISP

### Different Philosophies

**LISP**: Give the programmer maximum control
- Explicit iteration or recursion
- Programmer manages search strategy
- Macros customize the language

**Prolog**: Let the system figure it out
- Automatic backtracking search
- Pattern matching and unification built-in
- Declarative specification

### Complementary Strengths

**LISP excelled at**:
- Complex data transformations
- Custom control flow
- Metaprogramming
- Systems with complex state

**Prolog excelled at**:
- Search problems
- Pattern matching
- Database-like queries
- Constraint satisfaction

### Cultural Divide

American AI (MIT, Stanford, CMU) was LISP territory. European AI (Edinburgh, Marseille, Imperial College) embraced Prolog. Japan's Fifth Generation project briefly made Prolog glamorous worldwide.

## Modern Prolog

### Continuing Development

Prolog remains active:

**SWI-Prolog**: Open-source, widely used for teaching and research
**SICStus Prolog**: Commercial, industrial-strength
**GNU Prolog**: Free, efficient native code
**Tau Prolog**: JavaScript implementation for web

### Constraint Logic Programming

Modern Prolog often includes constraint solving:

```prolog
% Using CLP(FD) - Constraint Logic Programming over Finite Domains
:- use_module(library(clpfd)).

% Sum of digits from 1 to N equals S
sum_digits(N, S) :-
    N #>= 1,
    X #= N * (N + 1) / 2,
    S #= X.
```

### Logic Programming Ideas Elsewhere

Even where Prolog isn't used, its ideas appear:
- SQL's declarative queries
- Regular expressions as pattern matching
- Datalog in modern data systems
- Unification in type inference

## Key Takeaways

- Logic programming reversed the traditional approach: describe what you want, not how to compute it
- Prolog (1972) by Colmerauer and Kowalski was the main logic programming language
- Programs were facts and rules; execution meant proving queries via unification and backtracking
- Prolog excelled at parsing, expert systems, constraint satisfaction, and database-like queries
- Japan's Fifth Generation project (1982) bet heavily on Prolog, spurring international competition
- LISP and Prolog represented different philosophies: programmer control vs. declarative specification
- Logic programming ideas influenced SQL, type systems, and modern constraint solvers

## Further Reading

- Clocksin, William & Mellish, Christopher. *Programming in Prolog* (5th ed., 2003) - Classic textbook
- Kowalski, Robert. "Logic for Problem Solving" (1979) - Theoretical foundations
- Sterling, Leon & Shapiro, Ehud. *The Art of Prolog* (2nd ed., 1994) - Advanced techniques
- Bratko, Ivan. *Prolog Programming for Artificial Intelligence* (4th ed., 2011) - AI applications

---
*Estimated reading time: 9 minutes*
