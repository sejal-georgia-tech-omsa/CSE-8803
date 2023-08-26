# Module 2

## Topic 1

### Lesson 1: One-Hot Encoding

- Why numerical text representation?
  - the goal of NLP is to be able to design algorithms to allow computers to "understand" natural language in order to perform some task
  - computers are good with numbers, so how do we convert text data to numerical data that can be used in a model?

- Representing Words 
  - every word can be represented by a vector of 0 except for one position that has a vlaue of 1
  - Example: this is a simple sentence
    - "this" -> [1, 0, 0, 0, 0]
    - "is" -> [0, 1, 0, 0, 0]
    - "a" -> [0, 0, 1, 0, 0]
    - "simple" -> [0, 0, 0, 1, 0]
    - "sentence" -> [0, 0, 0, 0, 1]

- From Word to Document Representation

    ```python
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder

    text = "This is a simple sentence"
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(text.split())
    # binary encode
    onehot_encoder = OneHotEncoder()
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))
    onehot_encoded.toarray()
    ```
- One-Hot Encoder
  - in a corpus with a vocabulary $V$ of size $d$ (number of unique words or dimensions), a word $w$ is represented as a vector $X$ of size $d$ such as:
    - $X_i^w = 1$ if $\text{idx}(w) = i$
    - $X_i^w = 0$ otherwise
  - a document can be represented as matrix of size $n$ x $d$, where $n$ is the number of words in the document, or a single vector of dimension $d$, with multiple valus of 1 where the words from the vocabulary are present
    - Document: $D$ = *this is a sentence*
    - Vocabulary: $V$ - [*aardvark*, ..., *this*, ..., *is*, ..., *a*, ..., *sentence*, ..., *zyther*]
    - One Hot Encoding: $X^D$ = [0, ..., 1, ..., 1, ..., 1, ..., 1, ..., 0]

- Advantages and Disadvantages of One-Hot Encoding
  - Advantages
    - simple and easy to implement
  - Disadvantages
    - every word is represented as a vector of the size of the vocabulary: not scalable for a large vocabulary (100,000 words)
    - high dimensional sparse matrix which can be memory & computationally expensive
    - every word is represented independently: there is no notion of similarity / meaning in one-hot encoding
      - all the vectors are orthogonal
      - example: despite the words "good" and "great" carrying similar meaning and the word "bad" carrying the opposite meaning, the dot product is 0

## Topic 2

### Lesson 1: Bag of Words (BoW) Frequency Counting

- Bag of Words Model
  - represent each **document** as a **bag of words**, ignoring words' ordering for simplicity
  - represent a document as a column vector $X$ of word counts
  - the bag of words is a fixed-length representation, which consists of a vector of word counts:
    - Document: $D$ = *it was the best of times, it was the worst of times*
    - Vocabulary: $V$ = [*aardvark*, ..., *it*, ..., *best*, ..., *times*, ..., *zyther*]
    - Bag of Words: $X$ = [0, ..., 2, ..., 1, ..., 2, ..., 0]
  - the size of $X$ is 1 x $d$ where $d$ is the size of the vocabulary
  - a collection of $n$ documents is represented with a matrix of size $n$ x $d$

- Code Snippet Implementation of BoW

    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    my_text = ["it was the best of times, it was the worst of times"]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(my_text)
    print(X.toarray())
    vectorizer.get_feature_names()
    ```
    ```
    [[1 2 2 2 2 2 1]]
    ['best', 'it', 'of', 'the', 'times', 'was', 'worst']
    ```
    ```

- Advantages and Disadvantages of BoW
  - Advantages
    - simple and easy to implement
  - Disadvantages
    - every document is represented as a vector of the size of the vocabulary: not scalable for a large vocabulary (100,000) words
    - high dimensional sparse matrix which can be memory & computationally expensive
    - the order of words is disregarded and thus, the meaning coming from the context is not captured


## Topic 3

### Lesson 1: Term Frequency - Inverse Document Frequency (TF-IDF)

- Why do we need TF-IDF?
  - we learned that the bag-of-words approach does not provide a logical importance for words
    - for example: "this is the NLP class"
      - all the words have the same importance here, according to BoW: "the" is as important as "NLP"
  - TF-IDF will help us assign more logical importance to a vector of words for each document

- What is TF-IDF and when to use it?
  - a word's importance score in a document, among $N$ documents
  - everywhere you use "word count", you can likely use TF-IDF


- BoW Example
  - Vocabulary $V$ = [this, is, a, sample, another, example]
  - Document 1: "this is a sample"
  - Document 2: "this is another example"

    Document 1
    | Term | Term Count |
    | --- | --- |
    | this | 1 |
    | is | 1 |
    | a | 2 |
    | sample | 1 |

    Document 2
    | Term | Term Count |
    | --- | --- |
    | this | 1 |
    | is | 1 |
    | another | 2 |
    | example | 3 |

    | Document | this | is | a | sample | another | example |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | 1 | 1 | 2 | 1 | 0 | 0 |
    | 2 | 1 | 1 | 0 | 0 | 2 | 3 |

- TF (Term Frequency)
  - the number of appearances of a term in a document
  - will be high if terms appear many times in this document
  
  $tf("this", d_1) = \frac{1}{5} = 0.2$  
  $tf("this", d_2) = \frac{1}{7} \approx 0.14$

- Inverse Document Frequency (IDF)
  - 
