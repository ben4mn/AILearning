# Semantic Networks and Frames

## Introduction

When psychologists studied human memory in the 1960s, they found evidence that concepts are organized in networks—related ideas connected by meaningful links. "Bird" connects to "can fly," "has feathers," and "is-a animal." "Robin" connects to "bird," inheriting its properties.

This insight inspired one of AI's most influential representation schemes: semantic networks. And when researchers realized networks weren't structured enough, they developed frames—packages of knowledge about typical situations and objects.

Together, semantic networks and frames defined how a generation of AI systems organized knowledge.

## Semantic Networks

### Origins

M. Ross Quillian introduced semantic networks in his 1968 PhD thesis on semantic memory. He was modeling how humans store and retrieve word meanings.

His insight: memory is a network of nodes (concepts) connected by links (relationships).

### Structure

A semantic network consists of:

**Nodes**: Represent concepts, objects, events
```
BIRD, ROBIN, RED, FLYING, WINGS
```

**Links**: Represent relationships between nodes
```
ROBIN --is-a--> BIRD
BIRD --has--> WINGS
ROBIN --color--> RED
```

### Visual Representation

```
                   ANIMAL
                      ↑
                   is-a
                      |
    CANARY ←──is-a── BIRD ──is-a─→ PENGUIN
       |               |              |
    color           has/can        color
       ↓               ↓              ↓
    YELLOW      [WINGS, FLY]       BLACK
```

### Common Link Types

**IS-A**: Category membership
```
ROBIN is-a BIRD
BIRD is-a ANIMAL
```

**HAS-A (PART-OF)**: Composition
```
BIRD has-a WING
BIRD has-a BEAK
```

**CAN**: Capabilities
```
BIRD can FLY
FISH can SWIM
```

**PROPERTY**: Attributes
```
ROBIN color RED
ELEPHANT size LARGE
```

### Inheritance

A key feature was property inheritance. If you ask "Can a robin fly?":

1. Check ROBIN for "can fly" — not found
2. Follow is-a link to BIRD
3. Check BIRD for "can fly" — found: YES
4. Conclude: ROBIN can fly (inherited from BIRD)

This allowed efficient storage—general properties stored once at higher levels, inherited by all subconcepts.

### Quillian's Experiments

Quillian conducted experiments showing that retrieval time in human memory correlated with network distance. Verifying "A canary can sing" (direct property) was faster than "A canary can fly" (inherited from bird).

This supported the psychological reality of network organization.

### Limitations

Semantic networks had problems:

**Vague semantics**: What exactly did links mean? Different researchers used "is-a" differently—sometimes for class membership, sometimes for subset relations.

**Limited expressivity**: How do you represent "All birds except penguins fly"? Or "Some birds are endangered"?

**No clear inference**: Beyond following links, what reasoning was allowed?

## Frames

### Origins

In 1974, Marvin Minsky published "A Framework for Representing Knowledge," proposing frames as a richer representation scheme.

Frames captured the idea that understanding involves matching situations to stereotypical patterns.

### The Frame Concept

A frame is a data structure representing a stereotypical situation:

```
FRAME: Restaurant
  SLOTS:
    type: [fast-food, casual, fine-dining]
    cuisine: [Italian, Chinese, Mexican, ...]
    typical-actors: [customer, waiter, chef]
    sequence-of-events: [enter, be-seated, order, eat, pay, leave]
    location: a building
    purpose: eating a meal
```

When you enter a restaurant, you activate this frame. It tells you what to expect, what roles people play, and what will happen.

### Slots and Fillers

Frames organized knowledge into **slots** (attributes) with **fillers** (values):

```
FRAME: Person
  SLOTS:
    name: [DEFAULT: unknown] TYPE: string
    age: [DEFAULT: adult] TYPE: number RANGE: 0-120
    gender: TYPE: {male, female, other}
    occupation: TYPE: Job-Frame
    mother: TYPE: Person-Frame
    father: TYPE: Person-Frame
```

### Defaults and Inheritance

Frames supported **default values**:
```
FRAME: Bird
  can-fly: DEFAULT = true
  has-feathers: true
  has-wings: true

FRAME: Penguin
  IS-A: Bird
  can-fly: false  -- override default
  habitat: Antarctic
```

When asking about a penguin, the system first checks Penguin, then inherits from Bird where values aren't specified.

### Procedural Attachment

Slots could have procedures attached:

**IF-NEEDED**: Run when slot value is requested
```
SLOT: age
  IF-NEEDED: compute from birthdate and current-date
```

**IF-ADDED**: Run when slot value changes
```
SLOT: salary
  IF-ADDED: update tax-bracket
```

**IF-REMOVED**: Run when slot value is removed

This mixed declarative knowledge with procedural computation.

### Example: Understanding a Story

Consider: "John went to a restaurant. He ordered a hamburger. He left a big tip."

The restaurant frame provides:
- John is the customer
- There's an implicit waiter (who received the tip)
- John was seated, then ordered, ate, and paid
- The hamburger was the meal
- The tip went to the waiter

All this "obvious" understanding comes from frame knowledge.

## Scripts

### Schank's Extension

Roger Schank at Yale extended frames into **scripts**—stereotypical sequences of events:

```
SCRIPT: Restaurant
  TRACK: Coffee-Shop

  ROLES: Customer (C), Waiter (W), Cook (K)
  PROPS: Tables, Menu, Food, Check, Money, Tip

  ENTRY CONDITIONS: C is hungry, C has money

  RESULTS: C is not hungry, C has less money
           Restaurant has more money

  SCENES:
    SCENE 1 - Entering
      C enters restaurant
      C looks for empty table
      C decides where to sit
      C goes to table
      C sits down

    SCENE 2 - Ordering
      W brings menu
      C reads menu
      C decides on order
      C signals W
      W comes to table
      C orders from W
      W goes to K
      W gives order to K

    SCENE 3 - Eating
      K prepares food
      W brings food to C
      C eats food

    SCENE 4 - Leaving
      W writes check
      W brings check to C
      C calculates tip
      C leaves tip on table
      C pays check
      C leaves restaurant
```

### Story Understanding

Schank's SAM (Script Applier Mechanism) used scripts to understand stories. Given:

"John went to a restaurant and ordered a hamburger. When the hamburger came it was burnt. John left."

SAM could answer:
- Q: Did John eat the hamburger? A: Probably not (he left when it came burnt)
- Q: Did John pay? A: Uncertain (story doesn't say, script expects it)
- Q: Who brought the hamburger? A: The waiter (from script)

### Limitations

Scripts were rigid:
- What about restaurants that don't match the script?
- How do you combine scripts for novel situations?
- The number of needed scripts seemed unbounded

## Frame Systems in Practice

### FRL (Frame Representation Language)

At MIT, Goldstein and Roberts developed FRL (1977), a practical frame language with:
- Inheritance hierarchies
- Procedural attachment
- Default reasoning

FRL influenced later commercial systems.

### KRL (Knowledge Representation Language)

Bobrow and Winograd's KRL (1977) at Xerox PARC explored:
- Multiple perspectives on objects
- Context-dependent interpretation
- Integration with procedures

KRL was influential but never widely used.

### Commercial Frame Systems

Frame ideas entered commercial AI:
- **KEE** (Knowledge Engineering Environment): Sophisticated frame system
- **ART** (Automated Reasoning Tool): Combined frames with rules
- **KL-ONE**: Developed formal frame-based "description logics"

## Influence on Object-Oriented Programming

Frames directly influenced object-oriented programming:

| Frame Concept | OOP Concept |
|--------------|-------------|
| Frame | Class |
| Slot | Instance variable |
| Default value | Default value |
| IS-A hierarchy | Inheritance |
| Procedural attachment | Methods |
| Instance | Object |

Languages like Smalltalk and later C++ and Java incorporated these ideas, though typically without defaults and with less emphasis on inheritance.

## Evolution

### Description Logics

In the 1980s, researchers formalized frame semantics, creating "description logics":
- Precise semantics for is-a and part-of
- Defined inference procedures
- Computational complexity analyzed

This led to OWL (Web Ontology Language) for the Semantic Web.

### Hybrid Systems

Modern knowledge representation often combines:
- Frames/ontologies for structure
- Rules for inference
- Neural embeddings for similarity
- Probabilistic methods for uncertainty

The pure frame approach gave way to hybrid architectures.

## Key Takeaways

- Semantic networks represented knowledge as nodes (concepts) connected by labeled links
- Quillian's work (1968) showed networks matched aspects of human memory organization
- Inheritance allowed efficient storage—properties defined once at general levels, inherited by specifics
- Frames (Minsky, 1974) organized knowledge into stereotypical structures with slots, defaults, and procedures
- Scripts (Schank) extended frames to capture stereotypical event sequences for story understanding
- Frame systems became practical tools: FRL, KRL, KEE, and influenced commercial AI
- Frames directly influenced object-oriented programming concepts
- These ideas evolved into formal description logics and modern knowledge representation

## Further Reading

- Minsky, Marvin. "A Framework for Representing Knowledge." In *The Psychology of Computer Vision*, ed. Patrick Winston (1975)
- Schank, Roger & Abelson, Robert. *Scripts, Plans, Goals, and Understanding* (1977)
- Quillian, M. Ross. "Semantic Memory." In *Semantic Information Processing*, ed. Marvin Minsky (1968)
- Brachman, Ronald. "What IS-A Is and Isn't." *IEEE Computer* (1983) - Classic critique of vague link semantics

---
*Estimated reading time: 9 minutes*
