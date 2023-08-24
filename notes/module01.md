# Module 1

## Topic 1

### Lesson 1: Class Overview

- Why Natural Language Processing?
  - texts and documents are everywhere
  - hundreds of languages in the world
  - primary information artifacts in our lives
  - large volumes of textual data

- Examples of Applications of NLP
  - establish authorship, authenticity; plagiarism detection
  - classification of genres for narratives
  - tone classification; sentiment analysis (online reviews, twitter, social media)
  - code: syntax analysis (e.g., find common bugs from students' answers)
  - machine translation (e.g., Google Translate)

- What makes NLP challenging?
  - interdisciplinary field that lies at the intersection of linguistics and machine learning
  - ambiguity at multiple levels in the human language:
    - lexical (word level) ambiguity-- different meanings of words
    - syntactic (sentence level) ambiguity-- different ways to parse the sentence
    - interpreting partial information-- how to interpret pronouns
    - contextual information-- context of the sentence may affect the meaning of the sentence

- What will you learn in the class?
  - preprocessing
    - how to clean texts and documents
    - tokenization
    - reducing the inflectional forms of a word
      - stemming
      - lemmatization
    - normalization
  - text representation
    - one hot encoding
    - BoW (bag of words)
    - TF-IDF
    - embeddings
  - overview of classification methods
    - Naive Bayes
    - Logistic Regression
    - SVM
    - Perceptron
    - Neural Network
  - overview of deep learning
    - convolutional neural network
    - recurrent neural network
    - long short-term memory (LSTM)
  - overview of topic modeling
    - principal component analysis
    - singular value decomposition
    - latent dirichlet allocation
  - overview of transformer models
    - bidirectional encoder representations from transformers
    - generative pre-trained transformer

- What deliverables are expected?
  - 4 homeworks [85%]
    - HW1: text preprocessing, classification introduction
    - HW2: classification methods, dimensionality reduction, SVD
    - HW3: deep learning
    - HW4: transformers and unsupervised models
  - 10 quizzes [15%]
    - mostly conceptual
    - multiple choice questions
    - available for a specific duration within a week
    - limited time to finish each quiz
    - mandatory to be taken

## Topic 2

### Lesson 1: Text Preprocessing Techniques

- NLP Terminology
  - **Corpus**: collection of text
    - e.g., Yelp Reviews, Wikipedia
  - **Syntax**: the grammatical structure of the text
  - **Syntactic parsing**: process of analyzing natural language with grammatical rules
  - **Semantics**: the meaning of text
  - **Tokenization**: splitting longer pieces of texts into smaller pieces (tokens)
    - example of splitting a sentence into words:  
      this is a simple sentence -> "this" "is" "a" "simple" "sentence"
  - **Vocabulary**: list of unique words
  - **Stop words**: commonly used words such as "the", "a", "an", "is", "are"... that don't contrivute a lot to the overall meaning
  - **N-Grams**: a consecutive sequence of n words (2-5 is common) in a text
    - 1-gram (unigram)
    - 2-gram (bigram)
    - 3-gram (trigram)
    - Example of bigrams:  
      this is a simple sentence --> "this is", "is a", "a simple", "simple sentence"

