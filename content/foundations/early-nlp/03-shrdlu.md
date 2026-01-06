# SHRDLU

## Introduction

If ELIZA showed how little was needed to create an illusion of understanding, SHRDLU showed how much was needed for genuine understanding—even in a tiny domain.

Created by Terry Winograd at MIT between 1968 and 1970, SHRDLU was a natural language understanding system that could discuss, reason about, and manipulate a simulated world of colored blocks on a table. Within its domain, SHRDLU could understand complex sentences, answer questions, follow commands, and explain its reasoning.

For a moment, it seemed like real progress on natural language understanding had arrived. But the lessons from SHRDLU would prove sobering: the techniques that worked in the blocks world didn't scale to the real world.

## The Blocks World

SHRDLU operated in a simulated "blocks world":

```
        ___
       |   |        ___
       |   |       /   \
   ____|   |____  |  ○  |
  |             | |_____|
  |  ___        |
  | |   |       |
  | |   |   ___ |
  |_|___|__|   ||
      ___|     ||
     |         ||
_____|___blue__|___________ table

Objects:
- Blocks (cubes of different colors)
- Pyramids (triangular shapes)
- A box (container for blocks)
- A hand (for picking up and moving)
```

This world was deliberately simple:
- Finite set of objects with known properties
- Clear spatial relationships (on, in, beside)
- Defined actions (pick up, put down, move to)
- No ambiguity about what exists

SHRDLU could parse natural language about this world, execute commands, and answer questions.

## Sample Conversations

Here's a famous excerpt from Winograd's demonstration:

```
Person: Pick up a big red block.
SHRDLU: OK. (does it)

Person: Grasp the pyramid.
SHRDLU: I DON'T UNDERSTAND WHICH PYRAMID YOU MEAN.

Person: Find a block which is taller than the one you are holding
        and put it into the box.
SHRDLU: BY "IT", I ASSUME YOU MEAN THE BLOCK WHICH IS TALLER
        THAN THE ONE I AM HOLDING.
SHRDLU: OK. (does it)

Person: What does the box contain?
SHRDLU: THE BLUE PYRAMID AND THE BLUE BLOCK.

Person: What is the pyramid supported by?
SHRDLU: THE BOX.

Person: How many blocks are not in the box?
SHRDLU: FOUR OF THEM.

Person: Is at least one of them narrower than the one I told you
        to pick up?
SHRDLU: YES, THE RED CUBE.

Person: Is it supported?
SHRDLU: YES, BY THE TABLE.

Person: Can the table pick up blocks?
SHRDLU: NO.

Person: Can a pyramid be supported by a block?
SHRDLU: YES.

Person: Can a pyramid support a pyramid?
SHRDLU: I DON'T KNOW.

Person: Stack up two pyramids.
SHRDLU: I CAN'T.

Person: Will you please stack up both of the red blocks
        and either a green cube or a pyramid?
SHRDLU: OK. (does it)

Person: Which cube is sitting on the table?
SHRDLU: THE LARGE GREEN ONE WHICH SUPPORTS THE RED PYRAMID.

Person: Is there a large block behind a pyramid?
SHRDLU: YES, THREE OF THEM: A LARGE RED ONE, A LARGE GREEN CUBE,
        AND THE BLUE ONE.

Person: Put a small one onto the green cube which supports
        a pyramid.
SHRDLU: OK. (does it)

Person: Had you touched any pyramid before you put the green one
        on the little cube?
SHRDLU: YES, THE GREEN ONE.

Person: When did you pick it up?
SHRDLU: WHILE I WAS STACKING UP THE RED CUBE, A LARGE RED BLOCK,
        AND A LARGE GREEN CUBE.

Person: Why?
SHRDLU: TO GET RID OF IT.

Person: Why did you do that?
SHRDLU: TO CLEAN OFF THE RED CUBE.

Person: Why did you clear off that cube?
SHRDLU: TO PUT IT ON A LARGE GREEN CUBE.

Person: Why did you do that?
SHRDLU: TO STACK UP THE RED CUBE, A LARGE RED BLOCK, AND A LARGE
        GREEN CUBE.

Person: Why did you do that?
SHRDLU: BECAUSE YOU ASKED ME TO.
```

This conversation demonstrates:
- Complex sentence parsing
- Reference resolution ("it," "that cube")
- Memory of past actions
- Goal-directed planning
- Causal reasoning
- Inference about physical constraints

## How SHRDLU Worked

SHRDLU integrated multiple AI components:

### Natural Language Parser

SHRDLU used a sophisticated parser (based on "systemic grammar") that analyzed sentences into structures representing meaning. Unlike shallow pattern matching, this was deep syntactic and semantic analysis.

The parser could handle:
- Complex noun phrases ("the big red block on top of the green cube")
- Relative clauses ("a block which is taller than...")
- Pronoun resolution ("it," "them")
- Conjunction and disjunction

### Knowledge Base

The system maintained a database of:
- Objects and their properties (color, size, shape)
- Current spatial relationships
- History of actions taken
- Definitions of object types

```lisp
;; Example of blocks world knowledge representation
(BLOCK B1 COLOR:RED SIZE:LARGE)
(BLOCK B2 COLOR:GREEN SIZE:SMALL)
(PYRAMID P1 COLOR:BLUE SIZE:MEDIUM)
(ON B1 TABLE)
(ON P1 B1)
(SUPPORTS TABLE B1)
(SUPPORTS B1 P1)
```

### Planner

When given commands, SHRDLU planned how to achieve them:
- Break goals into subgoals
- Check preconditions (can't pick up if hand is full)
- Handle obstacles (must move pyramid to access block beneath it)

This was a STRIPS-style planner integrated with language understanding.

### Inference Engine

SHRDLU could reason about its world:
- Answer questions by querying its knowledge base
- Explain its actions by tracing goal structures
- Check physical constraints (pyramids can't support things)
- Handle negation and quantifiers

## The Implementation

SHRDLU was written in Lisp and ran on a PDP-10 computer. It was sophisticated for its time—about 20,000 lines of code.

Key technical achievements:

**Integrated Architecture**: Unlike earlier systems that separated parsing from reasoning, SHRDLU integrated them. Semantic interpretation happened during parsing, not after.

**Procedural Semantics**: Word meanings were represented as procedures that did things, not just symbols. "Pick up" meant actually moving an object in the simulation.

**Context Sensitivity**: The parser used context (the current state of the world, the conversation history) to resolve ambiguity.

**Mixed-Initiative Dialogue**: SHRDLU could ask clarifying questions ("WHICH PYRAMID?") rather than just failing.

## Why SHRDLU Seemed Revolutionary

SHRDLU made a major impression:

**Genuine Understanding**: Unlike ELIZA, SHRDLU actually understood sentences. It correctly interpreted complex syntax, resolved references, and executed appropriate actions.

**Reasoning Ability**: SHRDLU could explain its actions, answer questions about the past, and make inferences about the world.

**Natural Dialogue**: Conversations with SHRDLU felt natural. It handled clarification, pronouns, and context like a competent partner.

**Integration**: SHRDLU showed that parsing, knowledge, planning, and inference could work together.

Winograd's 1972 MIT PhD thesis became widely read. SHRDLU seemed to vindicate the symbolic AI approach.

## Why SHRDLU Didn't Scale

The same Winograd who created SHRDLU later became one of its most insightful critics. The techniques that worked in blocks world didn't generalize.

### The Closed World

Blocks world was completely specified. Every object, property, and relationship was known. The real world is open—infinite objects, unknown relationships, uncertain information.

SHRDLU couldn't handle:
- "Move the block next to my coffee cup" (unknown object)
- "Pick up something interesting" (undefined property)
- "I'm not sure if there's a pyramid there" (uncertainty)

### Common Sense

SHRDLU knew blocks physics: pyramids can't support things, you can't pick up two things at once. But this knowledge was hand-coded.

The real world requires millions of common-sense facts. Hand-coding them all was impossible. Later projects like Cyc attempted this and found it endless.

### Ambiguity

Blocks world language was relatively unambiguous. Real language is full of ambiguity:
- Word senses (does "bank" mean financial or river?)
- Metaphor ("I'm drowning in work")
- Pragmatics (what does "Can you pass the salt?" really mean?)

SHRDLU's techniques couldn't handle this richness.

### Learning

SHRDLU didn't learn. Every fact, rule, and word meaning was programmed. Real understanding requires acquiring knowledge from experience.

### The "Toy Domain" Problem

SHRDLU worked because blocks world was a toy—simple, closed, formal. Many AI systems succeed on toys and fail on reality. SHRDLU established this pattern.

## Winograd's Reflection

Terry Winograd moved away from AI after SHRDLU. In papers and his 1986 book with Fernando Flores (*Understanding Computers and Cognition*), he critiqued the assumptions underlying SHRDLU:

**Rationalist Tradition**: SHRDLU assumed meaning could be captured in formal representations. Winograd came to see meaning as situated in human practices and relationships, not reducible to symbols.

**Representationalism**: The idea that understanding involves building internal models of the world was challenged. Maybe understanding is more about action than representation.

**Context**: SHRDLU handled context within its conversation. But real context includes social settings, cultural backgrounds, embodied experience—far beyond what SHRDLU modeled.

Winograd later worked on computer-supported collaborative work and the design of interactive systems, applying his insights differently.

## Legacy

SHRDLU's contributions endure:

**Benchmark**: Blocks world became a standard test domain for planning, robotics, and NLP.

**Integration**: The idea of integrating parsing with semantic interpretation influenced later systems.

**The Winograd Schema**: In 2011, the "Winograd Schema Challenge" was proposed—sentences requiring common-sense reasoning to interpret correctly. The challenge honors SHRDLU's creator while acknowledging what it couldn't do.

**Cautionary Tale**: SHRDLU shows both what symbolic AI can achieve and where it hits limits. It remains a touchstone for understanding AI's challenges.

## Key Takeaways

- SHRDLU (1968-1970) was Terry Winograd's natural language system that operated in a simulated blocks world
- It achieved genuine understanding within its domain: parsing complex sentences, reasoning, planning, and explaining
- SHRDLU integrated parsing, knowledge representation, planning, and inference in a unified system
- The approach didn't scale: real-world language involves open domains, common sense, ambiguity, and learning
- Winograd himself became a critic of the representationalist assumptions underlying SHRDLU
- SHRDLU remains influential as both a landmark achievement and a cautionary tale about toy domains

## Further Reading

- Winograd, Terry. *Understanding Natural Language* (1972) - The full technical description
- Winograd, Terry & Flores, Fernando. *Understanding Computers and Cognition* (1986) - Winograd's later critique
- Levesque, Hector, Davis, Ernest, & Morgenstern, Leora. "The Winograd Schema Challenge." *KR* (2012) - Modern benchmark inspired by SHRDLU's limitations
- Dreyfus, Hubert. *What Computers Can't Do* (1972) - Contemporary critique that cited SHRDLU

---
*Estimated reading time: 9 minutes*
