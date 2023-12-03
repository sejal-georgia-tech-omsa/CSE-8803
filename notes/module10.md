# Module 10

## Topic 1

### Lesson 1: Part-of-Speech (POS) Tagging

Part-of-Speech (POS) Tagging
- Assigning a part-of-speech to a word in a sentence
  - Words can have more than one POS
  - Essential in NLP
    - in machine translation, helps when we need to reworder words in a phrase
    - in text-to-speech, helps with the pronunciation of words because it recognizes the verb tense
    - utilized by Automatic Speech Recognition (ASR) systems to calculate the performance of the transcription algorithm

### Lesson 2: Named Entity Recognition (NER)

Named Entity Recognition (NER)
- Named Entity (NE): (abstract or physical) real-world objects that can be given a proper name
  - Examples: Georgia Tech, France, United Airlines
  - often a multi-world phrase
- Named Entity Recognition (NER): finding the part of a text containing a NE and attributing a tag to it
- Most common tags
  - PER (Person): "Marie Curie"
  - LOC (Location): "Atlanta"
  - ORG (Organization): "Georgia Tech"
  - GPE (Geo-Political Entity): "Boulder, Colorado"
  - Also includes non-entities: dates, times, prices
- Application
  - sentiment analysis
  - question answering: when a question asks about an entity in the sentence
  - information summarization: we can use the named entities in a text to extract important information
- Algorithms for NER
  - Sequence models: RNN and Transformer
  - Large language models: BERT (find-tuned)
  - NE Chunker classifier: uses MaxEnt classifier (maximum entropy classifier)
    - use POS tag outputs as input for the NER task
- NER challenges
  - Segmentation: unlike POS tagging, where each word is assigned a tag, NER requires finding entities which are often multi-word phrases
  - Text ambiguity: the meaning of a word/phrase can vary in different contexts
    - "Franklin" can be a name of a person or the name of a city in North Carolina, so should it be tagged as a person (PER) or a location (LOC)
- IOB
  - using IOB or BIO tagging, we will have one tag (person, location, etc.) per NE. We also specify the beginning of the NE (B), beteween or inside the NE (I), and outside any NE (O)

Named Entity Recognition (NER) - Transformer Models
- fine-tuning a transformer architecture with an attention mechanism
- using Keras, we can create a fine-tuned transformer and test it to get better performance for our example
- we can choose the datasets to train on according to the testing sample
- Steps for building and training a transformer model
  1. Preparing the dataset by creating tags for the NER task
     1. iob_labels = ["B", "I"]
     2. ner_labels = ["PER", "ORG", "LOC", "MISC"]
     3. Example of IOB tagging: [('Alex', 'B-PER'), ('is','O), ('going', 'O'), ('to', 'O'), ('Los','B-LOC'), ('Angeles','I-LOC')]
  2. Creating the transformer architecture for NER and training on the training dataset-- creating the Transformer block, Token and Position Embeddings
  3. Tokenizing the text and converting it to IDs for the model
  4. Compile and fit the model on training data
  5. Testing the results on the example text

---------------------------------------------

### Quiz 9

1. Which of the following is NOT an example of common NER (named entity recognition) tags?

> NNP (proper noun singular)

2. Which of the following is FALSE?

> Large language models like BERT can never be used for NER

3. John wants to implement a Part-of-Speech (POS) tagging model in PyTorch using a Recurrent Neural Network (RNN). Which of the following sequence model architectures is appropriate for the problem John is trying to solve?

> one to many

4. POS tagging, a common task in natural language processing, involves assigning each word in a text its corresponding grammatical category such as noun, verb, adjective, etc., based solely on the word itself without consider its context in the sentence

> False

5. In sequence labeling tasks, each input data point is considered independently, without any relation to previous or future data points

> False

