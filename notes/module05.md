# Module 5

## Topic 1

### Lesson 1: SVD and Co-occurrence Matrices

Motivating Example: Bag of Words representation
- there can be many features
- solution: dimension reduction

What is dimensionality reduction?
- the process of reducing random variables under consideration  
  - one can combine, transform, or select variables
  - one can use linear or non-linear operations

Intuition
- approximate a D-dimensional dataset using fewer dimensions
- first rotate the axes into a new space
- the highest order dimension captures the most variance in the original dataset
- the next dimension captures the next most variance, etc.

Singular Value Decomposition
- for a matrix $X_{nxd}$, where n is the number of instances and d is dimension: $X = U \Sigma V^T$
- where
  - $U_{nxm} \rightarrow$ unitary matrix $\rightarrow UU^T = I$
  - $\Sigma_{mxm} \rightarrow$ diagnomal matrix of singular values of X
  - $V_{mxd} \rightarrow$ unitary matrix $\rightarrow VV^T = I$
- m columns represent a dimension in a new latent space such that m column vectors are orthogonal to each other and ordered by the amount of variance in the dataset in each dimension. m could at most have d dimensions

Co-Occurrence Matrices
- the meaning of a word is defined by the words in its surroundings
- we define a context window as the number of words appearing around a center word
- we create a co-occurrence matrix as follows:
  - Step 1: go through each central word - context pair in the corpus (context window length is commonly anything between 1 and 5)
  - Step 2: in each iteration, udpate in the row of the count matrix corresponding to the central word by adding +1 in the columns corresponding to the context words
  - Step 3: repeat last 2 steps many times
  - Example: "it was the **best** of times, it was the worst of times" with a context window of 2
    - the words "was", "the", "of" and "times" appear in the context of the central word and central word "best" and get incremented by +1

Co-occurrence Matrix from the above example

| | it | was |the | best | of | times | worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| it | 0 | 2 | 2 | 0 | 1 | 1 | 0 |
| was | 2 | 0 | 2 | 1 | 0 | 1 | 1 |
| the | 2 | 2 | 0 | 1 | 2 | 0 | 1 |
| best | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| of | 1 | 0 | 2 | 1 | 0 | 2 | 1 |
| times | 1 | 1 | 0 | 1 | 2 | 0 | 1 |
| worst | 0 | 1 | 1 | 0 | 1 | 1 | 0 |

SVD on Co-Occurrence Matrices

- for a corpus with a vocabulary V of size d, the occurrence matrix has a size of dxd
- the size of the co-occurrence matrix increases with the vocabulary
- instead of keeping all dimensions, we can use the truncated SVD to keep only the top k singular values, for example 300
- the result is a least square approximation to the original co-occurrence matrix X
- size changes:
  - $X = dxd$
  - $U = dxm \rightarrow dxk$
  - $\Sigma = mxm \rightarrow kxk$
  - $V = mxd \rightarrow kxd$

Dense Word Embeddings
- each row of U is a k-dimensional representation of each word w in the corpus that best preserves the variance
- generally, we keep the top k dimensions, which can range from 50 to 500
- this produces dense vectors for word representations while taking into consideration the word contexts which carry meaning

Advantages of Dense Word Embeddings
- denoising: low-order dimensions may represent unimportant information
- truncation may help the models generalize better to unseen data
- having a smaller number of dimensions may make it easier for classifiers to properly weight the dimensions for the task
- dense models may do better at capturing higher order co-occurrence
- dense vectors tend to work better in word similarity
- one word similarity method is cosine similarity between two-word embeddings $w$ and $v$

$$\text{cosine}(\bar v, \bar w) = \frac{\bar v • \bar w}{|\bar v||\bar w|} = \frac{\sum_{i=1}^N v_i w_i}{\sqrt{\sum_{i=1}^N v_i^2}\sqrt{\sum_{i=1}^N w_i^2}}$$

## Topic 2

### Lesson 1: Glove (Global Vectors for Word Representation)

GloVe Model
- GloVe stands for global vectors where global refers to global statistics of corpus and vectors are representation of words
- GloVe uses statistics of word occcurrences in a corpus as the primary source of information
- GloVe model combines two widwly adopted approaches for training word vectors
  - Global matrix factorization
  - Window-based methods

Co-Occurrence Matrix
- for a corpus of vocbulary $V$ of size $d$, the co-occurrence matrix is a symmetrical matrix of size $dxd$
- $X_{ij}$: the number of times word $j$ occurs in the context of the word $i$ after defining a window size
- $X_i = \sum_k X_{ik}$: summation over all the words which occur in the context of the word $i$
- $P_{ij} = \frac{X_{ij}}{X_i}: P$ is co-occurrence probability where $P_{ij}$ is the probability of word $j$ occurring in the context of word $i$

Example

| | it | was |the | best | of | times | worst |
| --- | --- | --- | --- | --- | --- | --- | --- |
| it | 0 | 2 | 2 | 0 | 1 | 1 | 0 |
| was | 2 | 0 | 2 | 1 | 0 | 1 | 1 |
| the | 2 | 2 | 0 | 1 | 2 | 0 | 1 |
| best | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| of | 1 | 0 | 2 | 1 | 0 | 2 | 1 |
| times | 1 | 1 | 0 | 1 | 2 | 0 | 1 |
| worst | 0 | 1 | 1 | 0 | 1 | 1 | 0 |

- $X_{i=0,j=1} = 2$
- $X_{i=0} = 6$
- $P_{i=0,j=1} = \frac{2}{6} = 0.33$

GloVe Cost Function
- GloVe suggests finding the relationship between two words in terms of probability rather than occurrence counts
- GloVe looks to find vectors $w_i$ and $w_j$ such as $w_i^T w_j = \log{(P_{ij})} = \log{(\frac{X_{ij}}{X_i})}$
- $\log{(X_i)}$ is independent of word $j$ and can be represented as a bias $b_i$
- Adding a bias term to restore the symmetry for vector $w_j$, we get:  
$$w_i^T w_j + b_i + b_j = \log{(X_{ij})}$$

GloVe Cost Function
- A weighted least squares is used as a cost function for GloVe model:  
$$J = \sum_{ij} f(X_{ij})(w_i^T w_j + b_i + b_j - \log{(X_{ij})})^2$$  
  - the function $f(x)$ is defined as $f(x) = (\frac{x}{x_{\text{max}}})^a$ if $x < x_{\text{max}}$ else 1  
  - weighting function $f(x)$ with $a = \frac{3}{4}$

GloVe Word Vectors
- the model is trained in batches of the training sample with optimizer to minimize the cost function and hence generate word and context vectors for each word
- each word in the corpus is thus represented by a dense vector of fixed size length
- the word vectors obtained by GloVe showcase that the meaning was captured in these vector representations through similarity as well as linear structure
- using Euclidean distance or cosine similarity between word vectors represents the linguistics or semantic similarity of the corresponding words
- the word vectors by Glove conserve linear substructures
- vector differences capture as much as possible the meaning specified by the two words
- for example, the underlying concept that differentiates man and woman, meaning gender, may be equivalently specified by other word pairs such as king and queen $w_{\text{man}} - w_{\text{woman}} = w_{\text{king}} - w_{\text{queen}}$

------------------------------------------

Quiz 4

1. Which of the following are advantages of GloVe?

> - [ ] 1. We can use corpus data directly almost without any preprocessing
> - [x] 2. It uses matrix factorization that can quickly generate different sized vectors
> - [x] 3. It relies on global statistics of word occurrence
> - [x] 4. It gives lower weight for highly requent word pairs thus preventing the meaningless stop words

2. Which of the following statements of GloVe is **TRUE**?

> GloVe representation captures the linguistics or semantic similarity of the words

3. Which of the following statements is **INCORRECT**?

> The size of co-occurrence matrix decreases with an increase in vocabulary size

4. Which is the following central idea underlying dimensionality reduction for text data?

> Capturing high variance in the data

5. Which of the following statements is **FALSE**?

> Dimensionality reduction can only use linear operations to reduce information loss