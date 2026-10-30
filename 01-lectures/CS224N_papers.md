## Week 1

### Word Vectors
Efficient Estimation of Word Representations in Vector Space (original word2vec paper)
Distributed Representations of Words and Phrases and their Compositionality (negative sampling paper)

### Word Vectors and Language Models
GloVe: Global Vectors for Word Representation (original GloVe paper)
Improving Distributional Similarity with Lessons Learned from Word Embeddings
Evaluation methods for unsupervised word embeddings

## Week 2

### Backpropagation and Neural Network Basics
matrix calculus notes
Review of differential calculus
CS231n notes on network architectures
CS231n notes on backprop
Derivatives, Backpropagation, and Vectorization
Learning Representations by Backpropagating Errors (seminal Rumelhart et al. backpropagation paper)

### Dependency Parsing
Incrementality in Deterministic Dependency Parsing
A Fast and Accurate Dependency Parser using Neural Networks
Dependency Parsing
Globally Normalized Transition-Based Neural Networks
Universal Stanford Dependencies: A cross-linguistic typology
Universal Dependencies website
Jurafsky & Martin Chapter 18

## Week 3

### Recurrent Neural Networks
N-gram Language Models (textbook chapter)
The Unreasonable Effectiveness of Recurrent Neural Networks (blog post overview)
Sequence Modeling: Recurrent and Recursive Neural Nets (Sections 10.1 and 10.2)
On Chomsky and the Two Cultures of Statistical Learning
Sequence Modeling: Recurrent and Recursive Neural Nets (Sections 10.3, 10.5, 10.7-10.12)
Learning long-term dependencies with gradient descent is difficult (one of the original vanishing gradient papers)
On the difficulty of training Recurrent Neural Networks (proof of vanishing gradient problem)
Vanishing Gradients Jupyter Notebook (demo for feedforward networks)
Understanding LSTM Networks (blog post overview)

### Sequence to Sequence Models and Machine Translation
Statistical Machine Translation slides, CS224N 2015 (lectures 2/3/4)
Statistical Machine Translation (book by Philipp Koehn)
BLEU (original paper)
Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)
Sequence Transduction with Recurrent Neural Networks (early seq2seq speech recognition paper)
Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)
Attention and Augmented Recurrent Neural Networks (blog post overview)
Massive Exploration of Neural Machine Translation Architectures (practical advice for hyperparameter choices)
Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models
Revisiting Character-Based Neural Machine Translation with Capacity and Compression

## Week 4

### Transformers
Attention Is All You Need
The Illustrated Transformer
Transformer (Google AI blog post)
Layer Normalization
Image Transformer
Music Transformer: Generating music with long-term structure
Jurafsky and Martin Chapter 10 (Transformers and Large Language Models)

## Week 5

### Pretraining
Aligning language models to follow instructions
Scaling Instruction-Finetuned Language Models
AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback
How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources
Direct Preference Optimization: Your Language Model is Secretly a Reward Model

### Post-training
Aligning language models to follow instructions
Scaling Instruction-Finetuned Language Models
AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback
How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources
Direct Preference Optimization: Your Language Model is Secretly a Reward Model

## Week 6

### Benchmarking and Evaluation
Mixed Precision Training
ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
LoRA: Low-Rank Adaptation of Large Language Models

### Efficient Neural Network Training
Mixed Precision Training
ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
LoRA: Low-Rank Adaptation of Large Language Models

## Week 7

### Speech Brain-Computer Interface
A high-performance speech neuroprosthesis
An accurate and rapidly calibrating speech neuroprosthesis
A high-performance neuroprosthesis for speech decoding and avatar control
Brain-Machine Interfaces (Principles of Neural Science chapter)

### Reasoning and Agents
Orca: Progressive Learning from Complex Explanation Traces of GPT-4
Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
ReAct: Synergizing Reasoning and Acting in Language Models
BAGEL: Bootstrapping Agents by Guiding Exploration with Language
WebArena: A Realistic Web Environment for Building Autonomous Agents

## Week 8

### Life after DPO
RewardBench: Evaluating Reward Models for Language Modeling
D2PO: Discriminator-Guided DPO with Response Evaluation Models
Social Choice for AI Alignment: Dealing with Diverse Human Feedback

### ConvNets, Tree Recursive Neural Networks and Constituency Parsing
Convolutional Neural Networks for Sentence Classification
Improving neural networks by preventing co-adaptation of feature detectors
A Convolutional Neural Network for Modelling Sentences
Parsing with Compositional Vector Grammars.
Constituency Parsing with a Self-Attentive Encoder

## Week 9

### An Introduction to Responsible NLP
Preface + Introduction chapter of the FairML book by Solon Barocas, Moritz Hardt, Arvind Narayanan
Introducing v0.5 of the AI Safety Benchmark from MLCommons

## Week 10


## Prompt 1

I compiled a list of recommended reading from CS224N. What are the top 5 papers? It should probably include Attention Is All You Need and other seminal papers.

## Answer 1

Looking at the syllabus from `CS224N_papers.md`, here are the **top 5 most seminal and impactful papers** that form the backbone of modern NLP and modern LLMs:

---

### 1. *Attention Is All You Need* (Vaswani et al., 2017)
*Listed under: Week 4 (Transformers)*
* **Why it's essential:** The foundation of all modern NLP, LLMs, Vision Transformers, and modern generative AI. It discarded recurrent architectures entirely in favor of multi-head self-attention, solving the parallelization bottleneck and enabling scaling to hundreds of billions of parameters.
* **Key concept to grasp:** Scaled dot-product attention, multi-head attention, and positional encodings.

---

### 2. *Efficient Estimation of Word Representations in Vector Space* (Mikolov et al., 2013 — Word2Vec)
*Listed under: Week 1 (Word Vectors)*
* **Why it's essential:** The paper that triggered the deep learning revolution in NLP. It introduced Skip-Gram and CBOW architectures, proving that dense geometric representations could capture rich semantic and syntactic relationships (`King - Man + Woman ≈ Queen`).
* **Key concept to grasp:** Contrastive learning / negative sampling and the distributional hypothesis.

---

### 3. *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau et al., 2014)
*Listed under: Week 3 (Sequence to Sequence Models)*
* **Why it's essential:** This is the paper that **invented the attention mechanism**. Prior to this work, Seq2Seq models compressed an entire source sentence into a single fixed-size bottleneck vector. Bahdanau et al. introduced soft alignment so the decoder could dynamically look back at all encoder states. Without this paper, the Transformer would not exist.
* **Key concept to grasp:** Additive (Bahdanau) attention and the information bottleneck of recurrent encoders.

---

### 4. *Aligning language models to follow instructions* (Ouyang et al., 2022 — InstructGPT)
*Listed under: Week 5 (Post-training)*
* **Why it's essential:** The paper behind ChatGPT. It transformed LLMs from raw next-token predictors (which often rambled or hallucinated) into helpful conversational assistants. It established the standard 3-stage alignment recipe: Pretraining $\to$ SFT $\to$ RLHF (Reward Modeling + PPO).
* **Key concept to grasp:** SFT limitations, Bradley-Terry preference modeling, and PPO with a KL penalty.

---

### 5. *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
*Listed under: Week 6 (Benchmarking & Efficient Training)*
* **Why it's essential:** The industry standard for Parameter-Efficient Fine-Tuning (PEFT). As foundation models scaled past tens of billions of parameters, full finetuning became computationally intractable for most practitioners. LoRA showed that freezing base weights and injecting low-rank decomposition matrices ($W + \Delta W = W + BA$) achieves parity with full finetuning with a fraction of the GPU memory and zero added inference latency.
* **Key concept to grasp:** Intrinsic rank of weight updates in deep networks and adapter-based tuning.

---

### Honorable Mention (Close 6th)
* **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (Rafailov et al., 2023): The successor to InstructGPT's PPO-based alignment, widely adopted in open-source model post-training (Llama 3, Mistral, Zephyr) for its simplicity and stability.




