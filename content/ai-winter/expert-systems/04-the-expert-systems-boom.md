# The Expert Systems Boom

## Introduction

While DENDRAL and MYCIN proved expert systems could work, they remained academic projects. The 1980s changed that. Expert systems became big business, attracting billions in investment, spawning hundreds of companies, and promising to transform how organizations made decisions.

At the center of this transformation was R1/XCON, a system at Digital Equipment Corporation that saved tens of millions of dollars annually—proof that AI could deliver real commercial value.

This is the story of the expert systems gold rush, its spectacular rise, and the seeds of its eventual decline.

## R1/XCON: The Breakthrough

### The Problem

Digital Equipment Corporation (DEC) was one of the world's largest computer companies, selling VAX minicomputers that businesses could configure with various options: memory sizes, disk drives, peripheral devices, software packages.

Each VAX order required configuration:
- Were all selected components compatible?
- What cables, adapters, and connectors were needed?
- How should components be physically arranged?
- Were power and cooling requirements met?

DEC employed hundreds of technical editors to configure orders. The process was slow, error-prone, and expensive. Misconfigured systems caused customer anger and costly corrections.

### Enter John McDermott

John McDermott, a computer scientist at Carnegie Mellon University, proposed building an expert system to automate configuration. DEC was skeptical but allowed a trial.

Starting in 1978, McDermott and a team built R1 (later renamed XCON for eXpert CONfigurer). It was written in OPS5, a production rule language.

### How R1/XCON Worked

R1/XCON used forward chaining with approximately 2,500 rules (eventually growing to over 10,000):

```
Rule: CONFIGURE-CPU-MEMORY
IF:   The current task is to configure memory
      AND the CPU type is VAX-11/780
      AND available memory slots = 16
      AND customer has ordered 8 MB memory
THEN: Assign memory boards to slots 0-3 of backplane 1
      Update available slots to 12
      Set next task to configure I/O
```

The system progressed through configuration stages:
1. Validate order for completeness
2. Configure CPU and memory
3. Assign devices to Unibus/Massbus
4. Generate floor layout
5. Create cable lists

### Commercial Success

R1/XCON was deployed in 1980 and rapidly proved its worth:

**Productivity**: One R1/XCON run replaced hours of human work
**Accuracy**: Error rates dropped dramatically
**Scalability**: The system could handle growing product complexity
**Speed**: Customers received configured orders faster

By 1986, DEC estimated R1/XCON saved $40 million annually. It processed 80,000 orders per year with 95-98% accuracy.

### Significance

R1/XCON was the first major commercial expert system success. It proved that:
- Expert systems could save real money
- They could handle complex industrial problems
- The investment in knowledge engineering paid off

This success story ignited the expert systems boom.

## The Gold Rush Begins

### Corporate AI Groups

Major corporations created AI departments:

**AT&T**: Expert systems for telecommunications
**General Electric**: Locomotive repair, turbine diagnostics
**Boeing**: Manufacturing process planning
**American Express**: Credit authorization
**Blue Cross/Blue Shield**: Medical claims processing

By the mid-1980s, hundreds of Fortune 500 companies had AI initiatives.

### AI Startups

Venture capital poured into AI startups:

**Teknowledge** (1981): Founded by Stanford AI graduates, developed expert system tools and consulting
**IntelliCorp** (1980): Created KEE (Knowledge Engineering Environment)
**Inference Corporation** (1982): Built ART (Automated Reasoning Tool)
**Carnegie Group** (1983): Spun out of CMU

Dozens of companies competed to sell expert system tools, development environments, and consulting services.

### LISP Machines

Special-purpose computers optimized for AI development became a significant market:

**Symbolics** (1980): Spun off from MIT's AI Lab
**Lisp Machines Inc. (LMI)** (1979): Another MIT spinoff
**Texas Instruments Explorer**: Major hardware vendor entering AI
**Xerox**: Sold Interlisp-D workstations

These machines cost $50,000-$150,000 each and sold thousands of units.

### The Fifth Generation Project

In 1982, Japan announced the Fifth Generation Computer Project—a $850 million, ten-year effort to build advanced AI systems. The announcement sent shockwaves through the US and Europe:

- Was Japan about to dominate AI?
- Would the US lose its technological edge?
- Should governments respond?

The result was dramatically increased funding:
- The US launched MCC (Microelectronics and Computer Technology Corporation)
- DARPA increased AI funding substantially
- The UK launched the Alvey Programme
- Europe launched ESPRIT

Competition drove investment to unprecedented levels.

## The Market Explodes

### Market Size

Expert systems spending grew exponentially:

| Year | Market Size (estimated) |
|------|------------------------|
| 1983 | $50 million |
| 1985 | $250 million |
| 1987 | $1 billion |

Some analysts projected $5-10 billion markets by the early 1990s.

### Applications Everywhere

Expert systems appeared in diverse domains:

**Manufacturing**:
- Process control
- Quality inspection
- Scheduling optimization

**Finance**:
- Credit authorization
- Fraud detection
- Portfolio management

**Medicine**:
- Diagnosis assistance
- Treatment planning
- Drug interaction checking

**Engineering**:
- Design automation
- Troubleshooting
- Configuration

**Military**:
- Battlefield management
- Equipment diagnosis
- Mission planning

### The Hype

Trade publications and business press celebrated AI:

*Business Week*: "Artificial Intelligence: It's Here"
*Fortune*: "A New Industrial Revolution"
*Time*: "Machines That Think"

Conferences drew thousands of attendees. Vendors made bold predictions. Consultants promised transformation.

## Expert System Tools

### Development Environments

Building expert systems became easier with sophisticated tools:

**KEE** (Knowledge Engineering Environment):
- Rich knowledge representation
- Multiple reasoning modes
- Graphics and simulation
- $50,000-$100,000 per license

**ART** (Automated Reasoning Tool):
- Powerful rule system
- Object-oriented features
- Forward and backward chaining
- High-end workstation required

**OPS5**:
- Production rule language
- Forward chaining
- Used for R1/XCON
- Less sophisticated but proven

### Expert System Shells

Shells provided pre-built inference engines that could be filled with domain knowledge:

**EMYCIN**: Empty MYCIN shell, influential early example
**M.1/S.1**: Teknowledge's commercial shells
**VP-Expert**: PC-based, accessible pricing
**CLIPS**: NASA's free shell, widely used

These reduced development effort but constrained system architecture.

### Hardware Trends

By the mid-1980s, hardware was shifting:

**LISP Machines**: Expensive, specialized, narrow market
**Workstations**: Sun, Apollo, HP offered cheaper alternatives
**PCs**: IBM PCs and compatibles became powerful enough for simple systems

The shift from specialized AI hardware to general-purpose machines would have significant consequences.

## Successful Applications

### XCON's Progeny

DEC's success with XCON spawned related systems:
- **XSEL**: Helped salespeople configure orders
- **XFL**: Fleet layout planning
- **XCLUSTER**: Cluster system configuration

Together, these systems demonstrated enterprise-scale AI deployment.

### American Express Authorizer's Assistant

AmEx built an expert system for credit authorization:
- Analyzed transaction patterns
- Detected potential fraud
- Guided authorization decisions

The system improved accuracy while handling millions of transactions.

### PROSPECTOR

SRI's PROSPECTOR advised on mineral exploration:
- Analyzed geological data
- Assessed likelihood of ore deposits
- Recommended drilling locations

PROSPECTOR made headlines when it helped locate a molybdenum deposit worth over $100 million—seemingly proving AI could make money directly.

### General Electric's DELTA/CATS

GE developed systems for diesel-electric locomotive repair:
- Diagnosed problems from symptoms
- Recommended repair procedures
- Captured expertise of senior engineers

These systems addressed the challenge of retiring expert maintenance workers.

## Cracks in the Foundation

### Maintenance Nightmares

As expert systems grew, they became hard to maintain:

- XCON grew from 2,500 rules to over 10,000
- Rule interactions became complex and unpredictable
- Adding new rules could break existing functionality
- Knowledge engineers became overwhelmed

The very success of systems like XCON revealed scalability problems.

### Knowledge Acquisition Bottleneck

The fundamental challenge remained: getting knowledge into systems was laborious:
- Expert interviews were time-consuming
- Experts couldn't always articulate their knowledge
- Knowledge changed, requiring constant updates
- Each domain required starting over

The bottleneck that DENDRAL had identified hadn't been solved.

### Hardware Changes

The specialized AI hardware market collapsed in the late 1980s:
- General-purpose workstations became cheaper and faster
- PCs could run modest expert systems
- LISP machines couldn't match price/performance trends

Symbolics, LMI, and other AI hardware vendors struggled or failed.

### Unmet Expectations

Many expert system projects failed to deliver:
- Domains proved harder than expected
- Benefits were less than projected
- Deployment and integration challenges emerged
- User resistance limited adoption

As disappointments accumulated, enthusiasm faded.

## Lessons from the Boom

### What Worked

**Narrow, well-defined domains**: Configuration, diagnosis, and classification tasks succeeded
**High-value decisions**: Where errors were expensive, investment paid off
**Available expertise**: Systems worked when domain experts could articulate knowledge
**Organizational commitment**: Long-term support enabled success

### What Didn't

**Overly broad scope**: Systems that tried to do too much failed
**Unrealistic expectations**: Promised benefits often weren't achieved
**Poor integration**: Standalone systems didn't fit workflows
**Neglected maintenance**: Systems degraded without ongoing support

### Market Evolution

The expert systems market didn't disappear—it evolved:
- Rules engines embedded in business applications
- Knowledge management systems
- Business rules management
- Eventually, machine learning approaches

The explicit AI branding faded, but the technology persisted.

## Key Takeaways

- R1/XCON at DEC proved expert systems could deliver massive commercial value, saving $40 million annually
- The 1980s saw an expert systems boom with billions in investment, hundreds of companies, and thousands of applications
- Japan's Fifth Generation Project spurred competitive responses from the US, UK, and Europe
- The market grew from $50 million in 1983 to over $1 billion by 1987
- Specialized LISP machines created a significant hardware market
- Successful applications included configuration, diagnosis, credit authorization, and manufacturing
- Cracks emerged: maintenance complexity, knowledge acquisition bottleneck, unmet expectations
- The boom would be followed by retrenchment in the late 1980s—the second AI winter

## Further Reading

- McDermott, John. "R1: A Rule-Based Configurer of Computer Systems." *Artificial Intelligence* 19 (1982)
- Feigenbaum, Edward & McCorduck, Pamela. *The Fifth Generation* (1983) - Captures boom-era optimism
- Crevier, Daniel. *AI: The Tumultuous History of the Search for Artificial Intelligence* (1993) - Chapter on the boom and bust
- Schank, Roger. "Where's the AI?" *AI Magazine* (1991) - Critical retrospective

---
*Estimated reading time: 9 minutes*
