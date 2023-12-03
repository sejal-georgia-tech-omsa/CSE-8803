# Module 11

## Topic 1

### Lesson 1: Topic Modeling - Latent Semantic Indexing

What is topic modeling?
- it is an unsupervised learning technique (no labels needed) to extract topics from documents and find documents that potentially share a common context
- technique is used to query documents that may not have all the keywords, but are still related to a topic
- we may retrieve documents that DON'T have the term "system", but they contain almost everything else ("data", "retrieval")

Latent Semantic Indexing (LSI)
- Main idea
  - map each **document** into some **concepts**
  - map each **term** into some **concepts**
- Concept: a set of terms, with weights
- for example, DBMS_concept
  - "data" (0.8)
  - "system" (0.5)
  - "retrieval" (0.6)

We need to construct the document-term matrix
- similar to the unigram bag of words

| | data | system | retrieval | lung | ear |
| --- | --- | --- | --- | --- | --- |
| doc1 | 1 | 1 | 1 | | |
| doc2 | 1 | 1 | 1 | | |
| doc3 | | | | 1 | 1 |
| doc4 | | | | 1 | 1 |

Convert Document-term Matrix into a Concept Matrix

- term-concept matrix

| | database concept | medical concept |
| --- | --- | --- |
| data | 1 | |
| system | 1 | |
| retrieval | 1 | |
| lung | | 1 |
| ear | | 1 |

- document-conceptr matrix

| | database concept | medical concept |
| --- | --- | --- |
| doc1 | 1 | |
| doc2 | 1 | |
| doc3 | | 1 |
| doc4 | | 1 |


Query Using the Concepts
Q: how to search, e.g., for "system"?
A: find the corresponding concept(s); and the corresponding documents

We Need SVD to create the concepts

- document-term matrix
- document-concept similarity matrix
- term-concept similarity matrix


Query on both document and term
- Document ('information', 'retrieval') will be retrieved by query('data') even though it does not contain 'data'

### Lesson 2: Topic Modeling - Latent Dirichlet Allocation (LDA

Latent Dirichlet Allocation (LDA)
- in LDA, we need to know the number of topics in advance
- let's say your documents are related to news documents, and you will assign 3 topics: sports, food, and economy
  - Topic 1, Topic 2, Topic 3
  - you can later label those topics based on the output of the algorithm
- now that you know the number of topics, the LDA core suggestion is
  - news documents with similar topics share common words
  - these topics can be discovered by finding a group of words occurring together in all documents
  - we need to create the document-topic matrix
  - we need to create the topic-word matrix

Documents-Topics Matrix (Distribution)
- for wach document, we need to find the probability of each topic
- LDA algorithm randomly starts assigning each word in our corpus to a topic
- Then, we can calculate in each document the word frequency for each topic, hence its probability

Topics-Words Matrix (Distribution)
- for each word, we can calculate the frequency of each word appearing in each topic and hence its probability

Probability of a document
- Probability of document = ...
- Dirichlet Distributions = documents-topics (triangle with topics on corners). M is the number of documents
- The multinomial is calculated based on the Dirichlet Distributions Documents-topics
- Multinomial Distributions to calculate the probability of each topic given the Dirichlet Distributions (documents-topics) for document j where z is the topic of word t in document j. N is the number of words in document j

Duty of Each Probability
- Let's day Document 1 has 5 words; now we need to use the Documents-topics Dirichlet Distribution to randomly assign a topic to each word
- For example, the chanc eof each word being assigned as t2 in document 1 is 80% and is higher than other topics
- Now by knowing the topics of all 5 words in documents using the $p(z_{j,t}|\theta_j)$, we can use the topics-words dirichlet distribution to randomly reassign a word to that topic
- For example, if word 1 (price) was randomly assigned to topic 1 in $p(z_{j,t}|\theta_j)$, and if we were to assign a new word to this topic, the chance of "stadium" or "football" would be the highest among others

Duty of Each Probability
- sequence
  - topics probability
  - words probability
  - randomly generating topics for words
  - randomly assigning words to selected topics
- we can generate M documents based on the predefined values for Dirichlet parameters $\alpha$ and $\beta$
- we can change the Dirichlet parameters $\alpha$ and $\beta$ and generate another M documents
- we can check what sets of Dirichlet parameters $\alpha$ and $\beta$ would re-generate documents closeset to the original ones
- to maximize the probability of generated document to be similar to the original one, we use Gibbs sampling
- this maximization tries to cluster similar words and cluster similar documents for a specific topic together

-------------------------------------------

### Quiz 10

1. LDA is a generative statistical model used in natural language processing to classify documents into predefined topics, where each topic is defined by a fixed set of words

> False

2. For representing the probability of a document in LDA, what kind of distribution is considered for document-topics representation?

> Dirichlet distribution

3. Which of the following statements is **FALSE**?

> - [ ] Topic modeling is an unsupervised learning technique to extract topics from documents
> - [ ] In LSI, main idea is to map each document and each term into some 'concepts'
> - [x] In LSI, concept is a set of terms with zero weights
> - [ ] SVD is used to create the concepts in LSI

4. Which of the following statements is **False**?

> - [ ] Both LSI and LDA are unsupervised learning algorithms
> - [x] In LSI, we need to know the number of concepts in advance
> - [ ] In LDA, we need to know the number of topics in advance
> - [ ] LDA can be used to query documents that may not have all the keywords but still related

5. Which of the following statements are **TRUE**? Select all that apply.

> - [ ] In LDA, there is no need to know the number of topics in advance
> - [x] In LDA, document-topics matrix contains the probability of each topic for each document
> - [x] In LDA, topics-word matrix contains the frequency (hence probability) of each word appearing in each topic

