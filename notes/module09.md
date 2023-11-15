# Module 9

## Topic 1

### Lesson 1: Transformers

Transformers History

The transformers below are all pre-trained as *language* models
- **Language model**: a model trained on a large amount of raw text using self-supervised learning, develops a statistical understanding of the language it has been trained on
- **Pre-trained model**: trained from scratch, fine-tuned using *transfer learning*
- **Self-supervised learning**: training a model on an unlabeled dataset

- 2017: transformers first introduced
- 2017: GPT (first pretrained model)
- 2018: BERT, a pretrained model for text summarization
- 2019: BART and T5 were the first pretrained models that use the original transformer architecture

Why were the language models created?
- It takes a long time to train transformers on a large corpus of data
- Fine-tuning a pre-trained model requires much less data and time than the pre-training process
- The training process is very lengthy and expensive, and creates a huge carbon footprint
- Pre-trained model is trained on a dataset similar to the fine-tuning dataset. Therefore, it has some knowledge about he dataset that we could use in the fine-tuning process.

Transformers
- A specific architecture ogigiriginally developed for Machine Translation and is used for myriads of NLP tasks:
  - sentiment analysis
  - named entity recognition (NER)
  - part-of-speech tagging (POS)
  - text summarization
  - filling in the blanks
  - ...

NLP Problems Refresher
- Machine Translation: translation from one natural languageinto another by means of a computerized system
- Sentiment Analysis: identifying emotion of a sentence / phrase (positive, negative, neutral, ...)
- Text Summarization: summarizing a text into a shorter version
- Fill in the blank: This course will teach you all about _____ models (mathematical, statistical)
- Named Entity Recognition (NER): a very important mmethod in NLP for detecting variety of entities in a text, e.g., people, places date, etc.
- Part-of-Speech Tagging (POS): assign a part-of-speech to a word in a sentence such as adjective, verb, etc.

Why Transformers?

- LSTMs had been the state-of-the-art models for NLP tasks, but they have various problems:
  - They are slow: tiems are fed to LSTMs one at a time (because they need the hidden state of the previous step to make predictions) and cna't be fed to the model in parallel
  - Usually need large memory
  - Transfer learning doesn't work on them. So, training on a new set of labeled data is needed every time (not practical)

### Lesson 2: Transformers *continued*

Structure of the Transformer Model
- **Encoder**: gets the input and creates a representation of it
- **Decoder**: gets the input representation along with the target output and generates the new output (outputs are generated one word at a time and is fed back to the decoder)
- **Representation**: a set of vectors produced by the encoder that can be read by the decoder to acquire information about the input. Embeddings of similar wrods are similar to each other.

Encoder and Decoder Can Be Used Separately in Structures:
- Encoder-only models: for tasks that require understanding the input
  - examples
    - *what is the meaning of this sentence?*
    - *fill in the blank with the right word*
  - good in the understanidng of input, extracting meaningful information, sequence classification, masked language modeling, and natural language understanding (NLU)
  - BERT, RoBERTa, ALBERT
  - bi-directional
- Decoder-only models: for generative tasks
  - examples
    - *guessing the next word in a sentence*
  - good in generating sequences, natural language generation (NLG)
  - GPT, GPT-2, GPT-3
  - uni-directional

Encoder and Decoder Models (Sequence-To-Sequence)
- Encoder-decoder models: for generative tasks that require input
- good for seq2seq tasks:
  - *translation*
  - *text summarization*
- BART and T5 use this structure

Transformer Architecture Encoder Stack
- contains encoder blocks (usually 6, here 3)
- receives its input from Embedding and Position Encoding
- encoder duty: mapping its input to a fixed-length vector, called a context vector
  - contains a crucial component: **self attention**
  - the context vector is fed to the decoder

Embedding + Positional Encoding
- embeddings come from pre-trained models
  - Word2Vec (Google)
  - GloVe (Stanford)
- ordering words is important to grasp the meaning of a sentence
- encoding (positional encoding) encodes the position of each word
- this information is added to the embedding result and is fed to teh encoder
- PE: positional encoding
- Pos: position of the word
- d: length of encoding vector (i.e. word2vec)

Encoder Block
- self-attention layer
  - computes the relationship between words in the input sequence
- skip connections
  - helps with exploding / vanishing gradient problems
- normalization layers
  - provides smoother gradients, faster training, and better generalization accuracy by normalizing the distribution of intermediate layers
- feed forward (linear) layer
  - consists of two linear layers with ReLU activation between them
  - main purpose is to transform the attention vectors into a form that is acceptable by the next encoder or decoder layer
  - each feed-forward layer has its own parameters:
    - $FFN(x) = ReLu(x\theta_1 + b_1)*\theta_2 + b_2$

Decoder Block
- self-attention layer
  - computes the relationship between words in the target sequence
- encoder-decoder attention layer
- normalization layer
- feed forward (linear) layer

Original Transformer Architecture
- the attention layer in the encoder has access to the whole input sequence since it needs to pay attention to other words (before an dafter) when creating a representation for a specific word
- the bottom attention layer in the decoder is a self-attention unit and has access to only the words translated so far
- the top attention layer is a regular attention unit and uses the output of the encoder to produce the most accurate output
- the transformer model is "auto regressive"; it receives the recently generated output as the new input for prediction

### Lesson 3: Transformers *continued*

Attention
- introduced to improve the performance of encoder-decoder models for machine translation
- attention allows making predictions about the output by paying "attention" to some parts of the input sequence (we deal with 2 sequences; an input and an output sequence)
- attention mechanism fixed the following issues of sequence-to-sequence models:
  - dealing with long-range dependencies (dependency among words separated by multiple senences) between words in a long sentence

Self-Attention
- when creating the representation for each word in a sequence, self-attention tells the model to pay attention to which words are in the same sequence
- example: in translating "I like exercising in the morning", when we want to create a respresentation for the word "exercising":
  - we'd want to look at the subject "I" because languages usually have different verb forms for different subjects
  - the other words don't matter much in the translation of this word
- words can have different meanings when placed next to other words
  - The *cat* drank the milk because **it** was hungry
  - The cat drank the *milk* because **it** was sweet
- self-attention extracts information about the meaning so that it can associate 'it' with the correct word

Self-Attention
- in the self-attention layer (in both encoder and decoder), the input is passed to three parameters: key, query, and value
- the weighted values are computed by a compatibility function of the query with the corresponding key
- key, query, and value matrices are parameters whose initial weights are small, randomly selected numbers
  - these parameters change as the model is trained on the dataset

How to Calculate Attention for Each Input
- each sequence (sentence) contains several words
- the attention score must be calculated for each word
  - first all the inputs need to be passed through embedding and positional encoding
  - key, query, and value matrices need to be initialized randomly (their values will be optimized as the model is trained)
    - the size of all matrices are the same and depends on the number of words in a sequence and the length of embedding
  - each vectorized word is multipled by key, query, and value matrices to create key, query and value representation vector

Calculate Attention Score
- this is wehre we can observe how self-attention is considering other words
- this is the equation to compute each word's attention score where $d_k$ is the embedding size, Q is the query matrix, K is the key matrix, and V is the value matrix
- $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

### Lesson 4: BERT and GPT

BERT: Bidirectional Encoder Representation from Transformers
- introduced to improve the previous uni-directional language models (standard LSTM or RNN); models that are trained on sentences using only one direction (left-to-right or right-to-left); uni-directional models are highly limited in processing the input to make predictions
- bi-directional models use information from both the past and future for better prediction, while uni-directional models only use past information
- good for tasks that need an understanding of language
  - e.g. natural machine translation
  - question answering
  - sentiment analysis
  - text summarization
- created by stacking up encoders
- trained on BooksCorpus (800M words) and English Wikipedia (2,500M words)
- using two unsupervised tasks for training:
  - masked language modeling (MLM)
  - next sentence prediction (NSP)

Masked Language Modeling (MLM)
- masking out words in the input and training the model to predict the masked words
- input: the man went to the [MASK_1]. He bought a [MASK_2] of milk
- labels: [MASK_1] = store, [MASK_2] = gallon
- BERT is a bidirectional model, i.e., to predict a masked word, the two words after and before the masked word are considered

Next Sentence Prediction (NSP)
- giving the model two sentences and training the model to learn the order of the sentences
- important because in many NLP tasks such as question answering an natural language inference (NLI), understanding the order of the sentences is crucial
- given two sentences, A and B, does B come after A?
  - example A
    - Sentence A: the man went to the store
    - Sentence B: he bought a gallon of milk
    - Label: IsNextSentence
  - example B
    - Sentence A: the man went to the store
    - Sentence B: penguins are flightless
    - Label: NotNextSentence

Example
- BERT model (using HuggingFace Transformer library)
  - fill-mask: filling in the blank with an appropriate word
  - bert-based-uncased: BERT model, pretrained on English language using a masked language modeling (MLM) objective
  - top_k: number of outputs
  - token: the predicted token id
  - token_str: the predicted token

GPT: Generative Pre-Training
- decoder-based model
- different versions: GPT-1, GPT-2, GPT-3, GPT-4
- used to generate human-like text
- auto-regressive model

------------------------------------------

### Quiz 8

1. BERT, a pre-trained transformer model used for NLP tasks, is a unidirectional model, meaning it only understands context from the left or the right of a given word, but not both simultaneously.

> False

2. Select all the unsupervised tasks for training BERT

> [ ] Named Entity Recognition (NER)
> [x] Masked Language Modeling (MLM)
> [x] Next Sentence Prediction (NSP)
> [ ] Part-of-Speech tagging (POS)

3. Select the problems that LSTM models suffer from

> [x] The sequential structure results in slow training
> [x] Large memory consumption
> [x] Incomptable with transfer learning
> [ ] Cannot be used for unsupervised learning

4. Which of the following statements regarding Transformers is **False**?

> [x] Encoder-only models are good for generative tasks
> [ ] Encoder-decoder models are good for text summarization
> [ ] Self-attention is used in both encoder and decoder
> [ ] Position information is added to the embedding result and is fed to the encoder

5. Which of the following statements is **False**?

> [ ] Encoder models predict one output element at a time based on the preceding elements
> [ ] Key, Query, and Value matrices in transformer models have the same shape
> [ ] Attention mechanism deals with long-range dependencies between words in a long sentence
> [x] When creating representation for a specific word, the attention layer in the encoder has access to only the words before that word

