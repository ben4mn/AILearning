# AI History Curriculum - Content Generation Instructions

## Overview
You are generating educational content for an AI History Learning Platform. This is a comprehensive curriculum covering the history of Artificial Intelligence from the 1940s to present day.

**Save ALL your work in this `/research` directory.**

## Your Output Files

Save your work as follows:
- `output/curriculum-full.json` - Complete curriculum with all topics, lessons, quizzes
- `output/lessons/{era-slug}/{topic-slug}/{order}-{lesson-slug}.md` - Individual lesson files
- `output/progress.md` - Track which topics you've completed

## Target Audience

- Technical professionals transitioning to AI/ML
- Upper undergraduate / graduate CS students
- Software developers wanting AI foundations
- **Assumes**: Basic programming knowledge, algebra, some calculus
- **Does NOT assume**: Prior ML/AI experience

---

## Lesson Requirements

Each lesson should be **1200-2000 words** of comprehensive content.

### Lesson Markdown Format

```markdown
# [Lesson Title]

## Introduction
Hook explaining why this matters and what we'll cover (2-3 paragraphs)

## [Main Section 1]
Core content with explanations, examples, historical context

### [Subsection if needed]
Detailed exploration of a specific concept

## [Main Section 2]
Continue building the concept

## [Main Section 3] (if needed)
Additional depth

## Key Takeaways
- 3-5 bullet points summarizing main ideas

## Further Reading
- Links to original papers, books, resources

---
*Estimated reading time: X minutes*
```

### Style Guidelines

- **Tone**: Conversational but authoritative (like a knowledgeable colleague explaining)
- **Voice**: Use first-person plural ("we'll explore", "let's understand")
- **Analogies**: Use analogies to familiar concepts when explaining abstract ideas
- **Context**: Include historical context and "why this matters" framing
- **Code**: Include Python code snippets when they aid understanding
- **Math**: Mathematical formulas should have intuitive explanations alongside them
- **Accuracy**: Cite specific years, publications, and researchers where relevant

---

## Quiz Requirements

Create **5-8 questions per topic** that test understanding.

### Question Distribution
- 60% should be answerable after careful reading of the lessons
- 40% should require deeper understanding or synthesis

### Question Quality Guidelines
- Test understanding, not just memorization
- Include plausible distractors (wrong answers that make sense if you misunderstand)
- Explanations should TEACH, not just confirm the answer
- Vary difficulty across questions

### Quiz Question JSON Format

```json
{
  "questionText": "Why was X significant for the development of AI?",
  "questionType": "multiple_choice",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correctAnswer": "Option C",
  "explanation": "Option C is correct because... This concept is important because it shows how...",
  "questionOrder": 1
}
```

---

## Topics to Generate (27 total)

### Era 1: foundations (1940s-1960s)
| # | Slug | Title |
|---|------|-------|
| 1 | turing-test | The Turing Test |
| 2 | perceptrons | Perceptrons & Early Neural Networks |
| 3 | dartmouth-conference | The Dartmouth Conference (1956) |
| 4 | symbolic-ai | Symbolic AI & Logic |
| 5 | early-nlp | Early Natural Language Processing |

### Era 2: ai-winter (1970s-1980s)
| # | Slug | Title |
|---|------|-------|
| 6 | first-ai-winter | The First AI Winter |
| 7 | expert-systems | Expert Systems & MYCIN |
| 8 | lisp-prolog | LISP, Prolog & AI Languages |
| 9 | knowledge-representation | Knowledge Representation |
| 10 | second-ai-winter | The Second AI Winter |

### Era 3: ml-renaissance (1990s-2000s)
| # | Slug | Title |
|---|------|-------|
| 11 | statistical-nlp | Statistical NLP Revolution |
| 12 | svms-kernels | Support Vector Machines |
| 13 | decision-trees-ensembles | Decision Trees & Random Forests |
| 14 | backpropagation-revival | Backpropagation Rediscovered |
| 15 | early-deep-learning | Early Deep Learning Attempts |

### Era 4: deep-learning (2010s)
| # | Slug | Title |
|---|------|-------|
| 16 | deep-learning-breakthrough | The Deep Learning Breakthrough |
| 17 | cnns-imagenet | CNNs & ImageNet |
| 18 | rnns-lstms | RNNs & LSTMs |
| 19 | word-embeddings | Word Embeddings (Word2Vec, GloVe) |
| 20 | transformers-attention | Transformers & Attention |
| 21 | gans-generative | GANs & Generative Models |

### Era 5: modern-ai (2020s)
| # | Slug | Title |
|---|------|-------|
| 22 | large-language-models | Large Language Models |
| 23 | tokenization-embeddings | Tokenization & Embeddings Deep Dive |
| 24 | prompt-engineering | Prompt Engineering |
| 25 | rag-retrieval | RAG & Retrieval Systems |
| 26 | ai-agents | AI Agents & Tool Use |
| 27 | ai-safety-alignment | AI Safety & Alignment |

---

## Lessons Per Topic

Design **3-5 lessons per topic** that build progressively. Each lesson should be a complete learning unit.

### Example: Topic "turing-test"

| Order | Slug | Title | Type |
|-------|------|-------|------|
| 1 | who-was-alan-turing | Who Was Alan Turing? | content |
| 2 | computing-machinery-and-intelligence | Computing Machinery and Intelligence (1950) | content |
| 3 | imitation-game-explained | The Imitation Game Explained | content |
| 4 | critiques-and-alternatives | Critiques and Alternatives | content |

### Example: Topic "transformers-attention"

| Order | Slug | Title | Type |
|-------|------|-------|------|
| 1 | limitations-of-rnns | Limitations of RNNs and LSTMs | content |
| 2 | attention-is-all-you-need | Attention Is All You Need (2017) | content |
| 3 | self-attention-mechanism | Understanding Self-Attention | content |
| 4 | transformer-architecture | The Full Transformer Architecture | content |
| 5 | bert-gpt-era | BERT, GPT, and the Era of Pre-training | content |

---

## Mind Map Connections

Include a `connections` array in `curriculum-full.json` showing how topics relate to each other. This creates the visual mind map.

### Connection Types
- `leads_to` - Direct progression/influence
- `enabled` - One technology enabled another
- `preceded` - Came before chronologically
- `conceptual_link` - Conceptually related across time
- `influenced` - Influenced the development of

### Example Connections

```json
{
  "connections": [
    { "fromTopicSlug": "turing-test", "toTopicSlug": "perceptrons", "connectionType": "leads_to", "label": "Inspired" },
    { "fromTopicSlug": "perceptrons", "toTopicSlug": "first-ai-winter", "connectionType": "leads_to", "label": "Critique caused" },
    { "fromTopicSlug": "backpropagation-revival", "toTopicSlug": "deep-learning-breakthrough", "connectionType": "enabled", "label": "Made possible" },
    { "fromTopicSlug": "transformers-attention", "toTopicSlug": "large-language-models", "connectionType": "leads_to", "label": "Foundation for" },
    { "fromTopicSlug": "turing-test", "toTopicSlug": "large-language-models", "connectionType": "conceptual_link", "label": "Finally approaching" }
  ]
}
```

Create meaningful connections that show the flow of AI history. Aim for 40-60 connections total.

---

## curriculum-full.json Structure

Your final `output/curriculum-full.json` should follow this structure:

```json
{
  "version": "1.0.0",
  "metadata": {
    "title": "History of Artificial Intelligence",
    "description": "A comprehensive journey through AI evolution from the 1940s to today",
    "targetAudience": "Technical professionals with basic programming knowledge",
    "totalEstimatedHours": 15,
    "generatedAt": "2026-01-05"
  },
  "eras": [
    { "id": "foundations", "name": "Foundations (1940s-1960s)", "description": "The birth of computing and first ideas about machine intelligence", "color": "#3B82F6", "order": 1 },
    { "id": "ai-winter", "name": "AI Winter & Expert Systems (1970s-1980s)", "description": "Funding cuts, renewed optimism, and knowledge-based systems", "color": "#8B5CF6", "order": 2 },
    { "id": "ml-renaissance", "name": "ML Renaissance (1990s-2000s)", "description": "Statistical methods reshape the field", "color": "#10B981", "order": 3 },
    { "id": "deep-learning", "name": "Deep Learning Revolution (2010s)", "description": "Neural networks finally deliver on their promise", "color": "#F59E0B", "order": 4 },
    { "id": "modern-ai", "name": "Modern AI (2020s)", "description": "Large language models and the age of generative AI", "color": "#EF4444", "order": 5 }
  ],
  "topics": [
    {
      "slug": "turing-test",
      "title": "The Turing Test",
      "description": "Alan Turing's foundational question: Can machines think?",
      "era": "foundations",
      "linearOrder": 1,
      "icon": "brain",
      "estimatedMinutes": 25,
      "lessons": [
        {
          "slug": "who-was-alan-turing",
          "title": "Who Was Alan Turing?",
          "contentPath": "foundations/turing-test/01-who-was-alan-turing.md",
          "lessonOrder": 1,
          "lessonType": "content"
        }
      ],
      "quiz": {
        "title": "Turing Test Knowledge Check",
        "passingScore": 70,
        "isGate": true,
        "questions": [
          {
            "questionText": "In what year did Alan Turing publish 'Computing Machinery and Intelligence'?",
            "questionType": "multiple_choice",
            "options": ["1943", "1950", "1956", "1965"],
            "correctAnswer": "1950",
            "explanation": "Turing published his seminal paper in 1950...",
            "questionOrder": 1
          }
        ]
      }
    }
  ],
  "connections": [
    { "fromTopicSlug": "turing-test", "toTopicSlug": "perceptrons", "connectionType": "leads_to", "label": "Inspired" }
  ]
}
```

---

## Research Sources to Use

Prioritize authoritative sources:
- Original papers (arxiv, ACM Digital Library, IEEE)
- "Artificial Intelligence: A Modern Approach" by Russell & Norvig
- "The Master Algorithm" by Pedro Domingos
- "Superintelligence" by Nick Bostrom (for modern era)
- Stanford AI Index Report (for recent statistics)
- Interviews and talks by key researchers
- Wikipedia (for historical facts, verify with primary sources)

---

## How to Work

1. **Start with topic 1** (turing-test)
2. **Design its lessons** (3-5 lessons that build progressively)
3. **Write each lesson markdown file** to `output/lessons/{era}/{topic}/{order}-{slug}.md`
4. **Create 5-8 quiz questions** for the topic
5. **Add the complete topic** (with lessons and quiz) to `output/curriculum-full.json`
6. **Update `output/progress.md`** with completion status
7. **Move to next topic**
8. **Repeat until all 27 topics are complete**

### Progress Tracking

Create `output/progress.md` to track your work:

```markdown
# Content Generation Progress

## Completed Topics
- [x] 1. turing-test (5 lessons, 6 questions)
- [x] 2. perceptrons (4 lessons, 5 questions)

## In Progress
- [ ] 3. dartmouth-conference

## Remaining
- [ ] 4. symbolic-ai
- [ ] 5. early-nlp
...
```

---

## Quality Checklist

Before marking a topic complete, verify:

- [ ] All lessons are 1200-2000 words each
- [ ] Each lesson has clear learning objectives in the introduction
- [ ] Historical claims have dates and can be verified
- [ ] Technical concepts have both intuitive and precise explanations
- [ ] Code snippets (if any) are correct and well-commented
- [ ] At least 5 quiz questions covering the material
- [ ] Quiz explanations are educational (teach, don't just confirm)
- [ ] Content flows logically from one lesson to the next
- [ ] Estimated reading times are realistic (assume 200 words/minute)
- [ ] Topic is added to curriculum-full.json with all data
- [ ] Lesson files are saved in correct directory structure

---

## File Naming Convention

```
output/
  curriculum-full.json
  progress.md
  lessons/
    foundations/
      turing-test/
        01-who-was-alan-turing.md
        02-computing-machinery-and-intelligence.md
        03-imitation-game-explained.md
        04-critiques-and-alternatives.md
      perceptrons/
        01-mcculloch-pitts-neuron.md
        02-rosenblatt-perceptron.md
        03-perceptron-learning-algorithm.md
        04-minsky-papert-critique.md
    ai-winter/
      first-ai-winter/
        ...
    ml-renaissance/
      ...
    deep-learning/
      ...
    modern-ai/
      ...
```

---

## Begin!

Start by generating content for **Topic 1: turing-test**. Create its lessons, quiz questions, and add it to curriculum-full.json. Then proceed through all 27 topics in order.

Good luck! This curriculum will help thousands of people understand the fascinating history of AI.
