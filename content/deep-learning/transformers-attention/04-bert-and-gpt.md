# BERT and GPT: The Pretraining Revolution

## Introduction

The Transformer architecture, powerful as it was, originally required task-specific training from scratch. But in 2018, two models showed that Transformers could be pretrained on massive text corpora and then fine-tuned for any downstream task. GPT from OpenAI demonstrated autoregressive language modeling at scale. BERT from Google showed that bidirectional pretraining could capture context from both directions.

These models launched the modern era of NLP, where pretrained language models became the starting point for almost every task. The paradigm shift from task-specific training to "pretrain then fine-tune" democratized NLP: anyone could now build on models that had learned from billions of words.

In this lesson, we'll explore how BERT and GPT differ in their pretraining approaches, why these differences matter, and how they set the stage for the large language model revolution.

## GPT: Generative Pre-Training

OpenAI's GPT (Generative Pre-Training, 2018) used a Transformer decoder to perform language modeling: predict the next token given all previous tokens.

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens):
        # Get embeddings
        x = self.embedding(tokens) + self.pos_embedding(positions)

        # Apply decoder blocks with causal masking
        for block in self.decoder_blocks:
            x = block(x, causal_mask=True)

        # Predict next token at each position
        logits = self.output(x)
        return logits
```

The pretraining objective:

```python
def gpt_pretraining_loss(model, text):
    """
    Predict next token at each position
    """
    logits = model(text[:-1])  # Input: all but last token
    targets = text[1:]          # Target: all but first token

    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    return loss
```

This is classic language modeling. GPT learns to predict what comes next, implicitly learning syntax, semantics, and world knowledge.

## BERT: Bidirectional Representations

BERT (Bidirectional Encoder Representations from Transformers) from Google took a different approach. Instead of predicting left-to-right, BERT sees the entire sequence and learns to fill in blanked-out words.

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        self.segment_embedding = nn.Embedding(2, embed_dim)  # For sentence pairs

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.mlm_head = nn.Linear(embed_dim, vocab_size)  # Masked LM prediction

    def forward(self, tokens, segment_ids):
        x = self.embedding(tokens) + self.pos_embedding(positions) + \
            self.segment_embedding(segment_ids)

        # Apply encoder blocks (no causal masking - sees everything)
        for block in self.encoder_blocks:
            x = block(x)

        return x
```

### Masked Language Modeling (MLM)

BERT's primary pretraining task:

```python
def bert_mlm_loss(model, text):
    """
    Randomly mask 15% of tokens, predict the masked ones
    """
    # Create masked version
    masked_text, mask_positions, original_tokens = mask_tokens(text)

    # Get representations
    representations = model(masked_text)

    # Predict only at masked positions
    masked_representations = representations[mask_positions]
    predictions = model.mlm_head(masked_representations)

    loss = F.cross_entropy(predictions, original_tokens)
    return loss
```

The masking strategy:
- 80% of masked positions: Replace with [MASK]
- 10%: Replace with random token
- 10%: Keep original token

This prevents the model from only learning when it sees [MASK].

### Next Sentence Prediction (NSP)

BERT's secondary task:

```python
def bert_nsp_loss(model, sentence_a, sentence_b, is_next):
    """
    Predict if sentence_b follows sentence_a
    """
    # Combine sentences with [SEP] token
    combined = [CLS] + sentence_a + [SEP] + sentence_b + [SEP]
    segment_ids = [0]*len(sentence_a) + [1]*len(sentence_b)

    representations = model(combined, segment_ids)

    # Use [CLS] representation for classification
    cls_rep = representations[0]
    prediction = model.nsp_classifier(cls_rep)

    loss = F.binary_cross_entropy_with_logits(prediction, is_next)
    return loss
```

NSP teaches the model about sentence-level relationships. (Later work showed this might be less important than MLM.)

## Bidirectional vs Autoregressive

The fundamental difference:

```
GPT (Autoregressive):
"The cat sat on the [?]"
Can only use: "The cat sat on the" to predict "mat"

BERT (Bidirectional):
"The cat [MASK] on the mat"
Can use: "The cat ... on the mat" to predict "sat"
```

**GPT's advantage**: Natural for generation. Each position is predicted given its true context (what came before).

**BERT's advantage**: For understanding tasks, seeing both directions helps. "Bank" in "river bank" is understood by seeing "river" even if it comes after.

## Fine-Tuning

Both models are fine-tuned for downstream tasks:

### GPT Fine-Tuning

```python
class GPTForClassification(nn.Module):
    def __init__(self, pretrained_gpt, num_classes):
        super().__init__()
        self.gpt = pretrained_gpt
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        # Get GPT representations
        representations = self.gpt(tokens)

        # Use last token's representation for classification
        last_rep = representations[:, -1, :]

        return self.classifier(last_rep)
```

### BERT Fine-Tuning

```python
class BERTForClassification(nn.Module):
    def __init__(self, pretrained_bert, num_classes):
        super().__init__()
        self.bert = pretrained_bert
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens, segment_ids):
        representations = self.bert(tokens, segment_ids)

        # Use [CLS] token representation for classification
        cls_rep = representations[:, 0, :]

        return self.classifier(cls_rep)
```

For other tasks:
- **Named Entity Recognition**: Classify each token
- **Question Answering**: Predict start and end positions in passage
- **Text Generation**: (GPT only) Continue generating tokens

## Impact and Results

Both models achieved dramatic improvements:

**GPT-1 (2018)**:
- 12 layers, 768 dimensions, 117M parameters
- Trained on BooksCorpus (7,000 books)
- State-of-the-art on 9 of 12 NLP tasks

**BERT (2018)**:
- Base: 12 layers, 768 dimensions, 110M parameters
- Large: 24 layers, 1024 dimensions, 340M parameters
- Trained on Wikipedia + BooksCorpus (3.3B words)
- State-of-the-art on 11 NLP benchmarks simultaneously

BERT's bidirectional nature made it particularly dominant for understanding tasks (question answering, classification, NER). GPT's autoregressive nature made it better for generation.

## The Pretrain-Finetune Paradigm

These models established a new paradigm:

**Old approach**:
1. Design task-specific architecture
2. Train from scratch on task data
3. Repeat for each new task

**New approach**:
1. Pretrain large model on massive text corpus (once)
2. Fine-tune pretrained model on task data (quick, small dataset OK)
3. Achieve state-of-the-art with minimal task-specific engineering

Benefits:
- **Data efficiency**: Fine-tuning needs far less labeled data
- **Compute efficiency**: Pretraining is done once, amortized across tasks
- **Accessibility**: Anyone can fine-tune; few can pretrain
- **Transfer**: Knowledge transfers across tasks

## Evolution: GPT-2 and Beyond

The path forward became clear: scale up.

**GPT-2 (2019)**:
- 1.5 billion parameters
- Trained on 40GB of internet text (WebText)
- Emergent capabilities: zero-shot task performance
- Too dangerous to release? (Initially staged release)

**BERT variants**:
- RoBERTa: Better training (no NSP, more data, longer training)
- ALBERT: Parameter sharing for efficiency
- DistilBERT: Knowledge distillation for smaller models
- XLNet: Combines autoregressive and bidirectional

**Beyond**:
- GPT-3 (2020): 175B parameters, remarkable few-shot learning
- T5: Text-to-text framework, unified approach
- And eventually... GPT-4, Claude, LLaMA, and modern LLMs

## Choosing Between Approaches

```python
# When to use BERT-style (encoder, bidirectional):
use_cases_bert = [
    "Text classification",
    "Named entity recognition",
    "Question answering",
    "Semantic similarity",
    "Tasks requiring understanding"
]

# When to use GPT-style (decoder, autoregressive):
use_cases_gpt = [
    "Text generation",
    "Completion/suggestion",
    "Creative writing",
    "Dialogue systems",
    "Tasks requiring generation"
]

# When to use encoder-decoder (T5-style):
use_cases_enc_dec = [
    "Translation",
    "Summarization",
    "Sequence-to-sequence tasks"
]
```

## Legacy

BERT and GPT established that:
1. Transformers scale well with data and compute
2. Self-supervised pretraining captures rich language understanding
3. A single pretrained model can adapt to many tasks
4. Bigger models learn more, and capabilities emerge with scale

Every major language model since—including GPT-4, Claude, Gemini, and LLaMA—builds on these foundations. The pretrain-finetune paradigm (and now prompt-based adaptation) remains the dominant approach.

## Key Takeaways

- GPT uses autoregressive (left-to-right) language modeling, making it natural for text generation
- BERT uses masked language modeling with bidirectional context, excelling at understanding tasks
- Both established the pretrain-finetune paradigm: train once on massive data, adapt quickly to any task
- BERT's [CLS] token and GPT's last token provide sequence-level representations for classification
- The success of these models demonstrated that scale (more data, more parameters) consistently improves capabilities
- These architectures directly led to modern LLMs like GPT-4 and Claude

## Further Reading

- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training" (GPT)
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)

---
*Estimated reading time: 11 minutes*
