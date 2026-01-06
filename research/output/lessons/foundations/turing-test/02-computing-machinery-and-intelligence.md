# Computing Machinery and Intelligence (1950)

## Introduction

In October 1950, the philosophical journal *Mind* published a paper that would become one of the most cited and debated works in the history of artificial intelligence. "Computing Machinery and Intelligence" by Alan Turing opened with a deceptively simple question: "Can machines think?"

But Turing immediately recognized that this question was problematic. The words "machine" and "think" were too vague and loaded with assumptions to permit a meaningful answer. So he proposed something revolutionary: replace the unanswerable philosophical question with a concrete, operational test. Instead of asking whether machines can think, we should ask whether machines can do something specific—something that, if a human did it, we would readily accept as evidence of thinking.

In this lesson, we'll examine Turing's seminal paper in detail, understanding both its arguments and its remarkable prescience about the future of computing.

## The Problem with "Can Machines Think?"

Turing begins his paper by acknowledging that the question "Can machines think?" seems straightforward but is actually deeply problematic. The difficulty lies in defining the terms:

**What is a "machine"?**
- Should we include biological machines (humans)?
- What about hybrid systems?
- Do we mean machines that exist today, or machines that could theoretically be built?

**What does it mean to "think"?**
- Is thinking the same as consciousness?
- Does it require understanding, or just behavior?
- How would we know if something "really" thinks versus merely appearing to think?

Rather than getting lost in these definitional debates—which Turing suspected would never be resolved—he proposed a pragmatic alternative. We should define a clear test and ask whether machines can pass it. If they can, the burden shifts to skeptics to explain why passing the test doesn't count as thinking.

This move from metaphysical questions to operational tests was methodologically brilliant. It allowed the field of AI to make progress without first resolving centuries-old philosophical debates about consciousness and mind.

## The Imitation Game

Turing introduced what he called "the imitation game," which has come to be known as the Turing Test. The original formulation went like this:

A human interrogator (C) communicates via text with two hidden participants: a human (B) and a machine (A). The interrogator's task is to determine which is the human and which is the machine. The machine's task is to fool the interrogator into making the wrong identification.

Crucially, communication is restricted to typed text. This eliminates physical appearance, voice, and other factors that would make the machine's identity immediately obvious. The question becomes: can a machine converse in a way that is indistinguishable from a human?

Turing wrote:

> "The question and answer method seems to be suitable for introducing almost any one of the fields of human endeavour that we wish to include."

This is key: the test is not limited to any particular domain. The interrogator can ask about poetry, mathematics, emotions, current events, personal experiences—anything. A machine that consistently fools interrogators would need to handle the full breadth of human discourse.

```
Example Interrogation (Turing's own example):

Interrogator: In the first line of your sonnet which reads "Shall I
              compare thee to a summer's day," would not "a spring
              day" do as well or better?
Witness:      It wouldn't scan.
Interrogator: How about "a winter's day." That would scan all right.
Witness:      Yes, but nobody wants to be compared to a winter's day.
Interrogator: Would you say Mr. Pickwick reminded you of Christmas?
Witness:      In a way.
Interrogator: Yet Christmas is a winter's day, and I do not think
              Mr. Pickwick would mind the comparison.
Witness:      I don't think you're serious. By a winter's day one
              means a typical winter's day, rather than a special
              one like Christmas.
```

Notice how this exchange requires cultural knowledge, emotional understanding, literary interpretation, and nuanced reasoning about language. A machine that could handle such conversations would be demonstrating something very much like human intelligence.

## The Digital Computer

Turing then provided a remarkably clear explanation of digital computers for his 1950 audience. He described three key components:

1. **Store (Memory)**: Holds information including both data and instructions
2. **Executive Unit (Processor)**: Carries out individual operations
3. **Control (Program Counter)**: Ensures instructions are executed in the correct order

He emphasized that digital computers are "universal machines"—they can imitate any other machine given the right program. This universality is crucial to his argument: if human thinking is computational, then a sufficiently programmed digital computer could replicate it.

Turing also addressed the distinction between discrete and continuous machines. While the human brain involves continuous physical processes, he argued that any continuous machine could be mimicked by a discrete one with sufficient precision. This claim remains debated today, but it was essential to his argument that digital computers were suitable candidates for implementing thinking.

## Objections and Replies

The heart of Turing's paper is a systematic examination of nine objections to the idea that machines could think. His responses reveal both his philosophical sophistication and his vision for how AI might develop.

### 1. The Theological Objection
*"Thinking is a function of man's immortal soul. God has not given souls to machines."*

Turing's reply: This objection assumes we know the limits of God's omnipotence. If God wanted to give a machine a soul, surely He could. More seriously, Turing notes this objection doesn't deserve deep engagement in a scientific paper.

### 2. The "Heads in the Sand" Objection
*"The consequences of machines thinking would be too dreadful. Let us hope they cannot."*

Turing dismisses this as wishful thinking with no argumentative force.

### 3. The Mathematical Objection
*"Gödel's theorem shows there are limits to what any formal system can prove. Machines are formal systems, so they have limits humans don't have."*

This is the most technically sophisticated objection. Turing had profound understanding of Gödel's incompleteness theorems (having invented similar results independently). His response: yes, any particular machine has limitations—but so does any particular human. There's no proof that humans can transcend all formal limitations.

### 4. The Argument from Consciousness
*"Machines don't have conscious experiences—they don't feel emotions or appreciate beauty."*

Turing acknowledges this is the strongest objection but points out we can never truly know another being's conscious experience. We accept that other humans are conscious based on their behavior and our similarity to them. If a machine behaved in ways indistinguishable from a conscious human, what grounds would we have for denying it consciousness?

### 5. Arguments from Various Disabilities
*"Machines can never do X"* (be kind, fall in love, learn from experience, make mistakes, etc.)

Turing notes these claims are usually based on limited experience with actual machines. He suggests most such limitations could be overcome with better programming and that some "disabilities" (like not making mistakes) might actually be advantages.

### 6. Lady Lovelace's Objection
*"Machines can only do what we program them to do. They cannot originate anything."*

This objection, attributed to Ada Lovelace (the first programmer), claims machines lack creativity. Turing's response is twofold: first, machines can surprise us—we don't always know what our programs will do. Second, learning machines could acquire abilities their programmers didn't explicitly give them.

### 7. Argument from Continuity in the Nervous System
*"The brain is continuous, not discrete like a digital computer."*

Turing argues that discrete systems can approximate continuous ones to any desired precision. If behavior is what matters, the underlying substrate shouldn't matter.

### 8. The Argument from Informality of Behaviour
*"Humans don't follow fixed rules. Our behavior is too flexible and context-dependent."*

Turing suggests this might be an illusion—we might be following rules too complex to easily identify. Or machines might learn context-dependent rules through experience.

### 9. The Argument from Extrasensory Perception
*"What about telepathy and clairvoyance? Machines can't do that."*

Remarkably, Turing took ESP seriously (based on then-recent experiments). He suggested putting the machine in a "telepathy-proof room" to ensure fair testing. This section now reads as quaint, but it shows Turing's willingness to address all objections, however unusual.

## Learning Machines

The final section of Turing's paper is perhaps the most prescient. Rather than trying to program all human knowledge into a machine, Turing proposed building a machine that could *learn*.

He drew an analogy to child development:

> "Instead of trying to produce a programme to simulate the adult mind, why not rather try to produce one which simulates the child's? If this were then subjected to an appropriate course of education one would obtain the adult brain."

This is the core insight behind modern machine learning. Turing envisioned:
- A simple initial program (the "child machine")
- A training process involving rewards and punishments
- Gradual development of complex behaviors through experience

He even anticipated that such training might produce unexpected results:

> "An important feature of a learning machine is that its teacher will often be very largely ignorant of quite what is going on inside."

This sounds remarkably like modern deep learning, where neural networks develop internal representations that their creators don't fully understand.

Turing predicted that by the year 2000, computers with 10^9 bits of storage (about 125 MB) could play the imitation game well enough to fool an average interrogator about 30% of the time in a five-minute test. While the timeline was optimistic, the direction was correct.

## Key Takeaways

- Turing replaced the vague question "Can machines think?" with the concrete "imitation game" test
- The test focuses on behavioral equivalence rather than metaphysical claims about consciousness
- Turing systematically addressed nine objections, many of which remain relevant today
- He envisioned learning machines that develop abilities through experience rather than explicit programming
- The paper's operational approach enabled AI research to proceed without resolving deep philosophical debates

## Further Reading

- Turing, Alan. "Computing Machinery and Intelligence." *Mind* 59, no. 236 (1950): 433-460. [Available online]
- Copeland, B. Jack. "The Turing Test." *Minds and Machines* 10 (2000): 519-539
- Moor, James H., ed. *The Turing Test: The Elusive Standard of Artificial Intelligence* (2003)
- French, Robert. "The Turing Test: The First Fifty Years." *Trends in Cognitive Sciences* 4, no. 3 (2000): 115-122

---
*Estimated reading time: 9 minutes*
