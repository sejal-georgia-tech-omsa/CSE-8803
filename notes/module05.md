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





------------------------------------------

### Quiz 3

1. WHich of the following is **NOT** a discriminative model?

> Naive Bayes

2. Logistic Regression is a "soft" classification algorithm, while the Perceptron and SVM are "hard" classification algorithms

> True

3. Which of the following statements of SVM is **FALSE**?

> The margin is $\frac{4}{||\theta||}$

4. Which of the following statements is **INCORRECT**?

> Logistic regression is a hard classification algorithm

5. Which one of the statements below is **CORRECT** about the dual form in the SVM model?

> We can use the kernel trick to help SVM handle non-linearly separable data



