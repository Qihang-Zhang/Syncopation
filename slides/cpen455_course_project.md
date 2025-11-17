---
title: "CPEN455 - Few-Shot Learning for Spam Detection"
---

# CPEN455 Project
## Few-Shot Learning for Spam Detection with LLMs

---

## Today's Agenda

1. Project Overview & Objectives
2. Getting Started - Setup & Quick Start
3. Understanding Bayesian Inverse Classification
4. Code Examples Walkthrough
5. KV Cache Mechanism
6. Tips for Success

---

## Project Overview

**Goal**: Use LLM to detect spam email via various techniques

**Techniques to Explore**: Bayesian inverse classification methods
- Zero-shot learning
- Naive prompting
- Full fine-tuning

**Dataset**: Email classification (spam vs. not spam)

---

## Learning Objectives

By completing this project, you are expected to:

- Explore Bayesian inverse classification
- Learn different prompting and fine-tuning strategies
- Gain hands-on experience with model evaluation and optimization
- Understand KV cache mechanisms in decoder-only transformers

---

## Quick Start Checklist

**1.Install UV Package Manager**

```bash
# Visit: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2.Install Dependencies**

```bash
uv sync
```

---

## Quick Start Checklist (cont.)

**3.Clone Autograder**

```bash
git clone git@github.com:DSL-Lab/CPEN455-Project-2025W1-Autograder.git autograder
```

:warning: **Windows users**: Clone first, then rename to `autograder`

**4.Verify Dataset**

Confirm datasets exist under:
```
autograder/cpen455_released_datasets/
```

---

## Running Examples

**1.Chatbot Example**

```bash
uv run -m examples.chatbot_example
```
Experiment with different prompts to see text generation capabilities.

**2.Bayes Inverse Zero Shot**
```bash
bash examples/bayes_inverse_zero_shot.sh
```
Baseline evaluation without extra training.

---

## Running Examples (cont.)

**3.Bayes Inverse Naive Prompting**
```bash
bash examples/bayes_inverse_naive_prompting.sh
```
Inject richer prompts at inference time.

**4.Bayes Inverse Full Finetune**
```bash
bash examples/bayes_inverse_full_finetune.sh
```
Fine-tune the model before evaluation.

---

## Bayesian Inverse Classification

### The Core Idea

Instead of training a discriminative classifier $P(Y|X)$, leverage the generative capabilities of LLMs by modeling:

$$P(X,Y) = P(X|Y) \cdot P(Y)$$

Then use Bayes' rule to get:

$$P(Y|X) = \frac{P(X,Y)}{P(X)} = \frac{P(X|Y) \cdot P(Y)}{\displaystyle\sum_{Y \in \mathcal{Y}} P(X|Y^{\prime}) \cdot P(Y^{\prime})}$$

---

## Bayesian Inverse Classification (cont.)

### LLM Application
<!-- TODO: make it can use a new line in the current env -->

$$
P_\theta(Y_\text{label}|X_{\leq i}) = \frac{P_{\theta}(X_{\leq i},Y_\text{label})}{P_{\theta}(X_{\leq i})}
$$

$$
=\frac{P_{\theta}(X_{\leq i}|Y_\text{label})P_{\theta}(Y_\text{label})}{\displaystyle\sum_{Y'}P_{\theta}(X_{\leq i}|Y')P_{\theta}(Y')}
=\frac{P_{\theta}(X_{\leq i},Y_\text{label})}{\displaystyle\sum_{Y'}P_{\theta}(X_{\leq i},Y')}
$$

---

## Bayesian Inverse Classification (cont.)

### Key Components

- $X_{\leq i}$: Input sequence (email content)
- $Y_\text{label}$: Label (spam or not spam)
- $\theta$: Pre-trained language model parameters
- $P_\theta(Y_\text{label}|X_{\leq i})$: Posterior probability
- $P_\theta(X_{\leq i}|Y_\text{label})$: Likelihood
- $P_\theta(Y_\text{label})$: Prior probability

---

## Example 1: Chatbot

### Purpose
Demonstrate general text generation capabilities

### What to Observe
- Model's ability of language understanding

### In Your Report
Document various prompts tried and model behaviors observed (5% running + 5% report = 10%)

---

## Example 2: Zero-Shot

### Purpose
Establish baseline performance without any training

### How It Works
- Direct classification via Bayesian inverse classification
- Model relies on pre-trained knowledge

### Expected Performance
Lower accuracy, but establishes baseline (10% total)

---

## Example 3: Naive Prompting

### Purpose
Improve performance with better prompts at inference

### How It Works
- Inject task-specific instructions
- Provide context about spam detection

### Expected Improvement
Better than zero-shot, still no training required (10% total)

---

## Example 4: Full Fine-tuning

### How It Works
- Train model on labeled spam/not-spam data
- Update all model parameters
- Requires more computational resources

### Expected Results
Highest performance among provided examples (10% total)

---

## Decoder-Only Transformers

### Architecture Overview
- Process tokens sequentially left-to-right
- Each token can only attend to previous tokens
- Examples: GPT models, LLaMA

### Key Characteristics
- Autoregressive generation
- Causal (unidirectional) attention
- Well-suited for text generation

---

## What is KV Cache?

During text generation, computing attention for all previous tokens repeatedly is wasteful.

### The Solution: KV Cache
Cache the Key (K) and Value (V) matrices from previous tokens:
- **Keys**: Computed from previous tokens
- **Values**: Computed from previous tokens
- **Query**: Only compute for the new token

This avoids recomputing K and V for tokens we've already processed!

---

## KV Cache: How It Works

### Without KV Cache
```
Generate token 1: Compute Q, K, V for token 1
Generate token 2: Compute Q, K, V for tokens 1, 2
Generate token 3: Compute Q, K, V for tokens 1, 2, 3
...
```
Time complexity: $O(n^2)$ where n is sequence length

---

## KV Cache: How It Works

### With KV Cache
```
Generate token 1: Compute & cache K, V for token 1
Generate token 2: Compute Q for token 2, reuse cached K, V
Generate token 3: Compute Q for token 3, reuse cached K, V
...
```
Time complexity: $O(n)$ - much faster! :rocket:

---

## Benefits of KV Cache

### :zap: Speed
- Reduces redundant computations
- Faster inference during generation

### :moneybag: Memory-Compute Tradeoff
- Trade memory for speed
- Significant speedup for long sequences

---

## Drawbacks of KV Cache

### :warning: Memory Consumption
- Grows linearly with sequence length
- Can be significant for long contexts
- Limits batch size

---

## KV Cache in This Codebase

### Implementation Location
Check the `model/` and `utils/` directories for:
- Model architecture definitions
- Attention mechanism implementation
- Cache management utilities

### What to Document
- Where KV cache is initialized
- How it's updated during generation

---

## Grading: Basic Parts (40%)

### Run Provided Examples
Each example worth 10% (5% for running, 5% for report):
1. Chatbot - try different prompts
2. Zero-shot baseline
3. Naive prompting
4. Full fine-tuning

**Note**: If you achieve ≥80% accuracy, you automatically get full 40% for basic parts! :tada:

---

## Grading: Advanced Parts (60%)

### Accuracy Milestones
- ≥ 80% accuracy: **5%**
- ≥ 85% accuracy: **Additional 5%**

### KV Cache Explanation: **20%**

### Leaderboard Competition: **30%**
- Relative ranking: $(1 - \frac{N}{\text{Total Students}}) \times 30$%

---

## Grading Breakdown

| Component | Weight | Requirement |
|:----------|:------:|:------------|
| Chatbot Example | 10% | Run & report |
| Zero Shot | 10% | Run & report |
| Naive Prompting | 10% | Run & report |
| Full Finetune | 10% | Run & report |
| Accuracy ≥ 80% | 5% | Any method |
| Accuracy ≥ 85% | 5% | Any method |
| KV Cache Explanation | 20% | Detailed report |
| Leaderboard | 30% | Relative ranking |

---

## Kaggle Submission

**1.Generate probabilities:**
```bash
uv run -m examples.save_prob_example
```

**2.Create Kaggle submission:**
```bash
uv run -m examples.prep_submission_kaggle \
  --input bayes_inverse_probs/test_dataset_probs.csv \
  --output kaggle_submission.csv
```

**3.Upload to [competition page](https://www.kaggle.com/t/7bd983ca8e064c9aa7f13cf1ecbdbf23)**

---

## Kaggle Notes

- **10 submissions per day limit**
- Public leaderboard: 70% of test data
- Final ranking: 100% of test data (revealed after deadline)
- Leaderboard is for reference only
- **Grading uses autograder results, not Kaggle!**

---

## Submission Requirements

### What to Submit (ZIP file)
1. **All source code**
2. **Report (report.pdf)** - NeurIPS format, max 4 pages
3. **Best model checkpoint** (if trained) in `examples/ckpts/`
4. **Interface**: `examples/save_prob_example.py` must work

### :warning: Critical Files
- Keep exactly ONE checkpoint file
- Ensure `bash autograder/auto_grader.sh` runs successfully
- Report must be named `report.pdf` in root directory

---

## Report Guidelines

### Format
- NeurIPS conference style
- Maximum 4 pages (excluding references/appendices)
- PDF format

### Suggested Structure
1. **Method** - Description, equations, figures
2. **Experiments** - Results, ablations, analysis
3. **Conclusion** - Findings, limitations, future work

---

## Report Content Tips

### Method Section
- Include a figure to explain your method (create yourself!)
- Use equations rigorously

### Experiments Section
- Ablation studies on design choices
- Training methods and special techniques
- Quantitative AND qualitative analysis
- Compare different approaches

---

## Other Methods to Explore

### Parameter-Efficient Fine-Tuning
- **Prefix-Tuning**: Train only prefix tokens
- **LoRA**: Low-rank adaptation of weights

### Data Augmentation
- **Data Synthesis**: Generate synthetic spam/non-spam emails
- Use powerful LLMs to augment training data

### Advanced Techniques
- **Ensemble Methods**: Combine multiple models
- **Prompt Engineering**: Advanced prompting strategies
- **Chain-of-Thought**: Multi-step reasoning

---

## Academic Integrity

### :white_check_mark: Allowed
- Use PyTorch functions
- Consult external resources (with citation)
- Use AI tools (with disclosure)

### :x: Violations (= ZERO grade)
- Code reuse without citation
- AI code generation without acknowledgment
- Manipulating test dataset for labels
- Fabricated results
- Sharing code/checkpoints with others

---

## Citation Requirements

### You MUST Cite
- Every dataset, paper, blog post, code snippet
- AI-assisted coding (e.g., ChatGPT, Copilot)
  - Include prompt-response summaries in appendix
  - Mention in main text
- Collaborators for informal feedback
- Any reused code bases

**Remember**: Proper attribution is not optional!

---

## Critical Warnings

### :no_entry: DO NOT
1. Change any code in `autograder/` folder
   - We use original code for grading
   - Changes may break grading → ZERO score

2. Include multiple checkpoint files
   - We cannot guarantee checking all
   - Include ONLY your best model

3. Forget to test autograder before submission
   - Run `bash autograder/auto_grader.sh`
   - Verify it completes successfully

---

## Resources

### Official Links
- [Project Repository](https://github.com/DSL-Lab/CPEN455-Project-2025W1)
- [Autograder Repository](https://github.com/DSL-Lab/CPEN455-Project-2025W1-Autograder)
- [Kaggle Competition](https://www.kaggle.com/t/7bd983ca8e064c9aa7f13cf1ecbdbf23)
- [UV Installation](https://docs.astral.sh/uv/getting-started/installation/)
- [NeurIPS Style Files](https://neurips.cc/Conferences/2023/PaperInformation/StyleFiles)

---

## Summary

### Key Takeaways
- LLMs can be adapted for classification via Bayesian methods
- Different strategies: zero-shot, prompting, fine-tuning
- KV cache is crucial for efficient inference
- Documentation and academic integrity matter
- Start early, experiment systematically

### Success Formula
**Understanding + Experimentation + Documentation = Success**

---

## Good Luck! :rocket:

- This is a learning experience
- Experiment and explore
- Document your journey
- Ask questions when stuck
- Have fun with it!

**Questions?**
