# Module 9

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

Transcripts

> Hi class. In this lecture, I will introduce one of the well-known sequence labeling tasks, name part of speech, or POS tag. And we will go over some programming practices. First, we need to know what is the main job of pos. Then we need to know how we are going to achieve these tasks. And what are the algorithms such as average perceptron, tiger or bird can help us with this task. In part-of-speech tagging, the goal is to label each word in a sentence with this grammatical role, such as noun, verb, adjective, etc. Pufs can be challenging task. Let's look at our example here. This is a sitting objective area where sittings grammatical role. Here is objective. Another, in another example like our sitting at my desk, the setting has the verb roll. This challenge is known as ambiguity, where many words in natural language can have multiple possible part of speech tags. So what are the benefits of finding out the role of each word in a sentence? Well, it will be very useful in machine translation, especially when we need to reorder words in a phrase. E.g. let's translate spacious car from English to French. By knowing that species is an injective, it would help us the order of, the order of eight when we translate into French, which will be on which specialists in text-to-speech, part of speech tagging will help us with the pronounciation of the wards because it recognizes the where tense e.g. I. Read the book every day. I read the book yesterday. We can also use part-of-speech tagging to evaluate the performance of the transcription algorithm, we need to tag the reference transcriptions or grounds, Ground Truth with POS tags. We also need to tag the output of the transcription algorithm with POS tags. Finally, we compare the POS tags of the reference transcriptions and the output of the transcription algorithm. Now, let's have a quick programming example of how we can use part-of-speech tagging using the well-known NLTK library. First, we need to import the NLTK library, the word tokenize method to tokenize the awards for the POS tagging task and the POS tag method. Based on the time this video is recorded, NLTK uses the average perceptron tagger to train is tagging model, which works very well. In tagging words. This simple algorithm works by vectorizing words using any well-known method that we already learned in our class, such as one-hot encoding, word2vec, or a customized one that works better for your POS tagging problem. Then we initialize random weights for each grammatical category, e.g. noun, verb, adjective, etc. For each word, a dot product operation is done against each provided grammatical category vector in the problem, which has the same size as the word vector. The grammatical category that leads to the highest score will be predicted as the tag. If it's wrong, when it's compared with the actual label, the weights will be updated. Here, one of our GTA students wanted me to use the sentence. Why not? Let's use it. This sentence is professor who's behind it. Travel from Atlanta to Paris. We are united a line to visit the Louvre Museum. We need to first token as the sentence and pass it into the POS algorithm. You can see that it tag professors as an NP, which is stands for proper nouns singular. And travel is tagged as VBD, standing for where past tense. Some of the main POS tags are n as noun, V as verb, ADJ as ejective, ADV as an adverb, PROGN as pronoun CEO, and j as conjunction, PRP, preposition, and d, t as determiner and etc. Now let's use bird as a transformer base deep learning model for the task of pos. I use the well-known Hugging Face library for this parishes, for this purpose. First, we need to define the model name for our deep learning pipeline, which is the bert model, fine-tune for the PUFs task. Then we need to pass it into auto model for token classification method. Finally, we use our classifier defined in our pipeline to apply part of speech to our example text. The bert model resulted well for the first word, which is professor, It is known, it is a non-word. And the chance of data would be in our training data was high. Now let's focus on my last name who's behind me. Certainly something quite new for this pre-trained model. Since we are using a pre-trained model, it is splitting the proper noun was behind it because the language model might not recognize this origin. This is why often the language models are fine tune for domain specific task. You may have a question about whether it can be used and encoder-decoder model instead of an encoder based models like bert? And the answer is yes, absolutely. Encoder-decoder models are typically used for tasks where the input and output sequence have different length, such as machine translation. In POS, the input sequence and the output sequence usually have the same length. So an encoder decoder can be an oval computing tasks for the POS. So in this lecture, we'll learn why part of speech tagging is needed and how we are going to do this task. I provided two algorithms that can achieve those. First one was average perceptron tiger, and the second one was Bert as an encoder base transformer model.

> Hi class. Today we will delve into the topic of named entity recognition or any ER, which is a common sequence labeling tasks in NLP. We will also cover some practical programming techniques for NER. To start, it is important to understand the purpose of named entity recognition. And then we will examine some algorithms that can assist in computing this task. These algorithms include any or chunker, which utilizes a maximum entropy model, and bird, a language model based on the encoder architecture. In the field of NER, the task is to identify and categorize named entities in a text document into a specific predefined categories, such as person names or organizations, locations, time expressions, et cetera. The purpose of this is to extract structured information from unstructured text data, which can then be utilized for various applications, including information retrieval, summarization, sentiment analysis, and question answering. The name entities in the tags or labels with tags indicating the category they belong to. The most common tags used in any ER or person for person names. Location, for geographical locations, organization for name of organizations, date for dates, and GPE for geopolitical entities. There may be other taxes well, depending on the specific NER, task. When it comes to algorithms for named entity recognition, several options exist. Sequence models such as RNN and transform. It can be used. Additionally, large language models like bert, which has been fine-tune, are also available. And now the algorithm using any ER is the any childcare classifier which operates using the maximum entry entropy or max and classifier. The difficulties in, in any ER include first segmentation part. While POS tagging assigns a tag to each individual word, NER involves locating entities that may consist of multiple boards. Second, ambiguity of texts. The interpretation of a word or phrase may change based on its contexts. For instance, Franklin could refer to a person's name or a CD in North Carolina, leading to the question of whether it should be tagged as a person or PUR PER or location as LOC. The IOB or BIO tagging system provides a way to tag name entities in an ER. The IOB tags allow to identify the beginning of a name entity, be the middle of a name entity I, or the outside of any name entity. This system allows to annotate named entities with a single tag per entity, while also preserving the boundary of the boundary inflammation of the entities which can be useful in certain NER, application, such as entity recognition and relation extraction. Let's have an example. Maddie rows behind you visited Florida last month. In this example, Maddie who is behind, is tagged as a person, entity or PR with the tag p, b, dash PER for the first word, and I dash PR for the second word. The word Florida is tagged as a location entity, LLC with a tag B dash LOC. The remaining words are tagged as 0, meaning they are outside any entity. In the NLTK library. We use any underscore chunk to recognize named entities. In the any chunking process, a classifier is trained on a set of annotated data and then uses this knowledge to identify entities in new tax. The classifier typically uses a machine-learning algorithm, such as the maximum entropy or max n classifier to make prediction based on the input features such as POS tags and what features CO and CO NLL is used to convert a tree structure generated from the name entity recognition NER tagging process into the IOB inside, outside beginning format, making it easy to compare and evaluate the NER results. The same example that one of our GT students wanted me to use for the POS tagging. Let's use it here for the NER task. Bear in mind, please. My students typically call me just Maddie. Here is the tax professor is bounded, traveled from Atlanta to Paris via United Airlines to visit the Louvre Museum. Here's an illustration of how the NLTK, any chunk function works with both binary equals true and binary equals false option. To perform named entity recognition, we can utilize the transformer model that has both an encoder and decoder. Using Keras, we can fine-tune the transformer and evaluate its performance with the test sample. We have the flexibility to choose a training dataset based on the test sample. The steps involved are as follows. First, creating tags for the NER task by preparing the dataset. Second, building the transformer architecture for NER and training on the training dataset, including the transformer block, token and position embeddings, three, tokenizing the tax and converting it to IDs for the model, for compiling and fitting the model on the training data. Finally, evaluating the results and the example tax. From the above output, we can see that the transformer model did not split entities, rules, boundaries, and luv, however, it was also not able to detect it as a named entity. The model performance can be improved by training the model and different dataset which might contain similar words. However, it is still tricky for a model to be able to recognize people's names every time. Transformer models are also not used often for the tasks such as NER. An encoding models like bert are preferred because bird is an encoder based architecture and leverages masking in the pre-training. And its output can be combined easily with another classification layer for token classification. In an encoder-decoder structure, we cannot tweak the individual outputs that easily. Therefore, training becomes more computationally expensive. Using the well-known Hugging Face library, we are going to use a bert model for the task of named entity recognition for the same example, tax, note that the bird NER model exists in PyTorch, hence be used there from underscore PT flag to import it in TensorFlow. This is output based on birth. The bert model separates the world was behind me into individual characters. As it only relies on pretrained data, may not be able to recognize unique proper noun that were not included in his training vocabulary. In such cases, especially as pre-training will be necessary for the NER task. In this lecture, we'll learn about NER task and then over some algorithms that can achieve solvent this task, e.g. any are chunkier and encoder decoder based transformer model, and finally a bert model. Note that the part of speech tagging is typically used in an ER problem to improve its performance.

### Quiz 8

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

