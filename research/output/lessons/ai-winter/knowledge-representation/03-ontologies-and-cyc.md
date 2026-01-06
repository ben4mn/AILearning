# Ontologies and Cyc

## Introduction

If the common sense knowledge problem was AI's biggest obstacle, Doug Lenat had a simple solution: just encode all of it. All the millions of facts that humans know about the world—physics, society, biology, language, culture—put them all in a database.

This was the Cyc project, launched in 1984 and still ongoing. It was the most ambitious attempt ever made to capture human knowledge. Cyc pushed the idea of formal ontologies—structured vocabularies defining the concepts and relationships in a domain—to its ultimate expression.

Whether Cyc succeeded, failed, or something in between reveals much about the challenges of knowledge representation.

## What Is an Ontology?

### The Philosophical Roots

In philosophy, ontology is the study of being—what exists and how things relate. In AI, an ontology became something more practical: a formal specification of concepts and their relationships.

### Components of an Ontology

A computational ontology typically includes:

**Concepts (Classes)**: Categories of things
```
Person, Animal, Vehicle, Event, Location
```

**Instances (Individuals)**: Specific examples
```
John_Smith, New_York_City, World_War_II
```

**Properties (Attributes)**: Characteristics
```
hasAge, hasColor, locatedIn
```

**Relations**: Connections between concepts
```
is-a, part-of, causes, before
```

**Axioms**: Rules and constraints
```
Every Person is-a Animal
Every mother is female
```

### Why Ontologies Matter

Ontologies provide:

**Shared vocabulary**: Everyone uses terms consistently
**Explicit assumptions**: Relationships are stated, not implicit
**Interoperability**: Different systems can exchange knowledge
**Reasoning foundation**: Inference becomes possible

## Early Ontology Work

### KL-ONE (1980s)

Ronald Brachman developed KL-ONE at BBN, one of the first knowledge representation systems with formal semantics:
- Clear definition of concept subsumption
- Automatic classification
- Tractable reasoning

KL-ONE spawned a family of "description logics" with various expressivity/tractability trade-offs.

### CYC Predecessors

Before Cyc, several projects encoded domain knowledge:
- **AM** (Lenat, 1976): Discovered mathematical concepts
- **EURISKO** (Lenat, 1983): Extended AM to other domains

These convinced Lenat that knowledge was key—and that encoding it was feasible if done systematically.

## The Cyc Project

### Origins

In 1984, Doug Lenat at MCC (Microelectronics and Computer Technology Corporation) launched Cyc with extraordinary ambition: encode all common sense knowledge that a typical person knows.

The name came from "encyclopedia"—and like an encyclopedia, Cyc would contain what educated people generally know.

### The Core Hypothesis

Lenat's hypothesis:
1. Common sense knowledge is finite (though large)
2. It can be encoded in logical form
3. Once encoded, it enables general intelligence
4. The encoding effort, though massive, would eventually complete

This was a bet that the knowledge acquisition bottleneck could be overcome through sheer sustained effort.

### Scale

Cyc's ambition was staggering:

**Initial projections**:
- 10 years of development
- $25+ million investment
- Team of knowledge enterers
- Millions of encoded facts

**What they built**:
- Over 40 years of development (and counting)
- Hundreds of person-years of effort
- Millions of assertions
- Hundreds of thousands of concepts

### Architecture

Cyc consists of several components:

**The Knowledge Base**: Millions of assertions in CycL (Cyc's representation language)

**Inference Engine**: Reasons over the KB to answer queries

**NL Interface**: Translates between English and CycL

**Tools**: For knowledge entry, browsing, and debugging

### CycL: The Representation Language

CycL is a first-order logic variant with extensions:

```cyc
; Concept definitions
(isa Dog BiologicalSpecies)
(genls Dog CanineAnimal)

; Facts
(isa Fido Dog)
(age Fido (YearsDuration 7))

; Rules
(implies
  (and (isa ?X Dog)
       (isa ?Y Dog)
       (parents ?X ?Y))
  (isa ?Y Dog))

; Default reasoning
(defaultTrue (isa ?X Dog) (quadruped ?X))
```

### Organization: Microtheories

Cyc uses "microtheories" (contexts) to organize knowledge:
- Knowledge valid in specific contexts
- Contradictions allowed across contexts
- Context inheritance

For example, "Sherlock Holmes lives at 221B Baker Street" is true in the Sherlock Holmes Fiction microtheory but not in Real World microtheory.

## What Cyc Knows

### Breadth of Coverage

Cyc covers vast domains:

**Physical world**:
- Objects have locations
- Liquids can be poured
- Fire burns
- Gravity pulls down

**Temporal reasoning**:
- Events have duration
- Causes precede effects
- Days follow nights

**Social knowledge**:
- People have jobs
- Marriage is a relationship
- Governments have laws

**Biological knowledge**:
- Animals eat
- Plants need light
- People are born and die

### Sample Knowledge

```cyc
; Physical knowledge
(implies
  (and (isa ?OBJ SolidTangibleThing)
       (not (physicallyContains ?SUPPORT ?OBJ)))
  (eventuallyResultsIn
    (releasingHold ?AGENT ?OBJ)
    (falling ?OBJ)))

; Social knowledge
(implies
  (and (guests ?EVENT ?GUEST)
       (isa ?EVENT BirthdayParty)
       (hasABirthdayDuring ?BPERSON ?EVENT))
  (isa ?BPERSON Person))

; Temporal knowledge
(implies
  (earlier ?E1 ?E2)
  (not (earlier ?E2 ?E1)))
```

### A Concrete Example

How does Cyc know that you can't eat a book?

```cyc
(genls Book InformationBearingThing)
(genls InformationBearingThing Artifact)
(isa Eating IngestingSomething)
(argIsa IngestingSomething 2 EdibleStuff)
(not (genls Artifact EdibleStuff))
; Therefore: Books are not EdibleStuff
; Therefore: You can't eat a book
```

## Success and Criticism

### What Cyc Achieved

**Scale**: The largest formal knowledge base ever built
**Longevity**: Sustained effort over decades
**Commercial use**: Applied in some industrial applications
**Research contributions**: Techniques for large-scale KB management

### The Criticisms

**Never complete**: After 40 years, common sense still isn't fully captured

**Brittleness**: Edge cases keep appearing; knowledge is always incomplete

**Effort underestimated**: The "10 year" project is now past 40 years

**Integration difficulty**: Using Cyc effectively requires expertise

**Competing approaches**: Machine learning increasingly outperforms hand-coded knowledge

### The Fundamental Challenge

The more you encode, the more exceptions you discover:

"Birds fly" → but not penguins, ostriches, birds with broken wings, dead birds, birds in cages, birds that are too young...

For every rule, endless qualifications. The knowledge keeps multiplying.

## Modern Ontologies

### Domain Ontologies

Rather than encoding everything, modern practice focuses on domains:

**Gene Ontology**: Biological genes and functions
**SNOMED CT**: Medical terms
**WordNet**: English word relationships
**Schema.org**: Web content markup

These are smaller, more focused, and more successful.

### The Semantic Web

Tim Berners-Lee's vision of a "Semantic Web" relied on ontologies:
- **RDF**: Resource Description Framework for data
- **RDFS**: RDF Schema for simple ontologies
- **OWL**: Web Ontology Language for rich ontologies

The Semantic Web hasn't fully arrived, but knowledge graphs (Google, Wikidata) use these technologies.

### Knowledge Graphs

Modern AI uses knowledge graphs:

```
(Albert_Einstein, birthPlace, Ulm)
(Ulm, country, Germany)
(Albert_Einstein, knownFor, Theory_of_Relativity)
```

Google's Knowledge Graph, Wikidata, and DBpedia encode millions of facts—though with less deep reasoning than Cyc attempted.

## Lessons from Cyc

### What We Learned

**Scale is harder than expected**: Even massive effort doesn't capture all common sense

**Maintenance is ongoing**: Knowledge changes; keeping it current is work

**Use matters**: Knowledge must integrate with applications to be useful

**Hybridization helps**: Combining formal knowledge with learned representations is promising

### The Ongoing Debate

Does AI need explicit knowledge representation?

**Pro formal knowledge**:
- Explainable reasoning
- Precise semantics
- Compositional generalization

**Pro learned representations**:
- Less manual effort
- Handles noise and ambiguity
- Scales with data

Modern systems increasingly combine both: knowledge graphs with neural embeddings, ontologies with machine learning.

## Key Takeaways

- Ontologies are formal specifications of concepts and relationships in a domain
- The Cyc project (1984-present) attempted to encode all common sense knowledge
- Cyc has grown to millions of assertions over 40+ years but remains incomplete
- CycL is Cyc's representation language, based on first-order logic with extensions
- Microtheories allow context-dependent knowledge and managed contradictions
- Critics note Cyc's incompleteness, brittleness, and the underestimated effort required
- Modern ontologies focus on specific domains: Gene Ontology, SNOMED, WordNet
- Knowledge graphs (Google, Wikidata) represent lightweight structured knowledge
- The field increasingly combines formal ontologies with machine learning

## Further Reading

- Lenat, Douglas & Guha, R.V. *Building Large Knowledge-Based Systems* (1990) - The Cyc rationale
- Lenat, Douglas. "Cyc: A Large-Scale Investment in Knowledge Infrastructure." *Communications of the ACM* (1995)
- Baader, Franz et al. *The Description Logic Handbook* (2nd ed., 2007) - Formal foundations
- Guarino, Nicola, ed. *Formal Ontology in Information Systems* (1998) - Ontology theory

---
*Estimated reading time: 9 minutes*
