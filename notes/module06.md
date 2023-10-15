# Module 6

## Topic 1

### Lesson 1: Neural Networks Forward Pass and Back Propagation

Inspiration from biological neurons
- Neurons: core components of the brain and the nervous system consisting of:
    1. Dendrites that collect information from other neurons
    2. An axon that generates outgoing spikes

Inputs >> Summation >> Activation (Sigmoid) >> $h(x)$
- $output = activation(x\theta+b)$

| Neuron | Activation Function |
| --- | --- |
| linear unit | $z$ |
| threshold / sign unit | $sign(z)$|
| sigmoid unit | $\frac{1}{1+\text{exp}(-z)}$ |
| rectified linear unit (ReLU) | $max(0,z)$ |
| tanh unit | $tanh(z)$ |

- can connect a bunch of these blocks together
- neural network regression
- neural network classification
- can increase the depth of each layer
- can add more hidden layers


Forward Pass
- $U_{11} = \sum_{i=0}^d x_i \theta_i = \theta_0 + \theta_1x_i + ... + \theta_d x_d$
- $O_{11} = \frac{1}{1+e^{-u_{i1}}}$
- in forward pass, we calculate all $u_{ij}$ and $o_{ij}$ values from the left to the right of the network

Backpropagation
- in backpropagation, we update all $\theta_i$ parameters from the right to the left of the network
- optimization can be done using iterative techniques such as gradient descent

## Topic 2

### Lesson 1: Word2Vec (CBOW and Skip-gram)

One-Hot Encoding Review
- the simplest word embedding is one-hot encoding
- example: one document
  > Apple and orange are fruit
- using one-hot encoding with common words such as "and" removed from the corpus
  - apple => [1, 0, 0, 0]
  - orange => [0, 1, 0, 0]
  - are => [0, 0, 0, 1]
  - fruit => [0, 0, 0, 1]

The Issues with One-Hot Encoding
- the size of each word vector equals the vocabulary size in our corpus
  - this can be huge if we have millions of words in our vocabulary
- a very long one-hot encoded vector is a waste of storage and computation
- the curse of dimensionality issue can emerge in this case of a very large vector
- if we have a new corpus, then the size of each word vector will be different, and the model we trained on the previous corpus will be useless in the case of transfer learning

Contextual Meaning of the Words
- if we use one-hot encoding, we cannot say "apple" and "fruit" share some common features as they are both fruits
- one-hot encoding is just a 0 and 1 embedding and does not consider the contextual meaning of the words
- no correlation between words that have similar meanings or usage

What do we want to achieve from word embedding?
- can we come up with a word embedding that can capture a numerical similarity value?
- similarity value of (apple and orange) == similarity value of (orange and apple)
- similarity value of (apple and orange) > similarity value of (apple and are)

First Algorithm: Continuous Bag of Words Model (CBOW)
- this algorithm will use neural networks to learn the underlying representation of words
- **one big caveat!!!**
  - a neural network is a supervised learning technique, and it needs labels
  - we need to develop a way to synthesize the labels from our corpus

Neighboring Words for CBOW (Label Creation)
- given the neighbors of a word, can we predict the center word?
    > apple and _____ are fruit
- given the context or neighbors of the blank, can I predict the blank by a window size (the window size is the hyper-parameter)?
- for simplicity, a window size of 1 for blank will be
    > apple, ____, are
  - we moved the common words (i.e., "and")
- in fact, we want to find the $P(orange|context)$, and we want to maximize this probability

Embedding every single word in the corpus using its context
- we are going to find the embedding representation of the word "orange"
- first, all of the words in the vocabulary need to be encoded using one-hot encoding
- each word will have $d$ dimensions (the size of the vocabulary)
- embedded word "orange" after applying softmax
- each value in the vector will be the probability score of the target word at that position

Second Algorithm: Skip-Gram Model
- given a center word, what could be the context (neighbors) of the center word?
  - **opposite of CBOW**
  > ___ orange ___ fruit
- given the center word (orange), can I predict the blanks by a window size (the window size is the hyperparameter)?
- in fact, we want to find the $P(context|orange)$, and we need to maximize this probability

CBOW or Skip-Gram Model?
1. CBOW can learn better **syntatic** relationships between words while Skip-gram is better in capturing better **semantic** relationships
   1. CBOW would learn that cats and cats are similar
   2. Skip-gram woudl learn that cat and dog are similar
2. CBOW is trained to predict (maximize the probability) a single word from a fixed window size of context words, while Skip-gram does the opposite and strives to predict several context words from a single input word
3. CBOW is much faster to train than Skip-gram