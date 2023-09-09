# Module 3

## Topic 1

### Lesson 1: Classification Introduction

Supervised Learning: two types of tasks
- regression: curve fitting
- classification: class estimation

Classification Example 1: Handwritten Digit Recognition
- represent input image as a vector $x \in \mathbb{R}^P784$
- learn a classifier $f(x)$ such that $f: x \rightarrow \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\}$

Classification Example 2: Spam Detection
- spam or not spam from email (related to NLP)

Regression Example 1: Apartment Rent Prediction
- suppose you are to move to Atlanta and you want to find the most reasonably priced apartment satisfying your needs:
  - square feet
  - number of bedrooms
  - distance to campus
  - etc.

Regression Example 2: Stock Price Prediction
- predict stock price over time

## Topic 2

### Lesson 1: Naive Bayes

Let's start with the math concept: Bayes Decision Rule
- $P(y|x)=\frac{P(x|y)P(y)}{P(x)}$
- posterior = likelihood * prior / normalization constant
- x is a document encoded (i.e., by BoW)
- y is the label of the document (i.e., document contains a positive or negative message)

Bayes Decision Rule
- prior: $p(y)$
- class conditional distribution: $p(x|y)$
- posterior probability of a test point:  
$q_i(x) := P(y=i|x) = \frac{P(x|y)P(y)}{P(x)}$
- bayes decision rule
  - if $q_i(x) > q_j(x)$, then $y=i$, otherwise $y=j$
  - OR if ratio $l(x) = \frac{P(x|y=i)}{P(x|y=j)} > \frac{P(y=j)}{P(y=i)}$, then $y=i$, otherwise $y=j$

What do people do in practice?
- Generative models
  - model prior and likelihood explicitly
  - "generative" means able to generate synthetic data points
  - Examples: Naive Bayes, Hidden Markov Models
- Discriminative models
  - Directly estimate the posterior probabilities
  - No need to model underlying prior and likelihood distributions
  - Examples: logistic regression, SVM, neural networks

Generative Model: Naive Bayes
- use bayes decision rule for classification  
$P(y|x) = \frac{P(x|y)P(y)}{P(x)}$
- but assume $p(x|y=1)$ is fully factorized
  - **dimensions (unique words) are independent**
- $p(x|y=1) = \Pi_{i=1}^d p(x_i|y=1)$
- or the variables corresponding to each dimension of the data are independent given the label

"Naive" Conditional Independence Assumption
- $P(y|x)=\frac{P(x|y)P(y)}{P(x)} = \frac{P(x,y)}{P(x)}$
- $P(x|y_{label=1})P(y_{label=1}) = P(x,y_{label=1})$ 
- $P(x_1|y_{label=1})P(x_2|y_{label=1})...P(x_d|y_{label=1})P(y_{label=1})$
- $P(y_{label=1})\pi_{i=1}^d P(x_i|y_{label=1})$

Example: Conditional Independence
- $P(y|x) = \frac{P(x|y)P(y)}{P(x)} = \frac{P(x,y)}{P(x)}$
- $\text{Vocabulary } V = [nice, give, us, this, is, ssn, information, job, a]$
- $P(document|y=positive)P(y=positive)=P(x=nice|y=positive)P(x=give|y=positive)....P(y=positive)$
- $P(document|y=negative)P(y=negative)=P(x=nice|y=negative)P(x=give|y=negative)...P(y=negative)$

How to represent the likelihood?
- $P(y|x)=\frac{P(x|y)P(y)}{P(x)}$
- $\text{Vocabulary } V = [nice, give, us, this, is, ssn, information, job, a]$
- A common distribution in NLP for Naive Bayes is the **Multinomial Distribution**
  - $P(x=nice|y=positive) = \frac{\text{count word nice in all documents with positive labels}}{\text{count all words with positive labels}}$
  - $P(y=positive) = \frac{\text{count \# positive documents}}{\text{count \# all documents}}$

Advantages and Disadvantags of Naive Bayes
- Advantages
  - simple and easy to implement
  - no training needed
  - good results in general
- Disadvantages
  - the position of the words in the document does not matter (BoW approach)
  - conditional independence

## Topic 3

### Lesson 1: Classification Evaluation

Classification Performance Confusion Matrix
- *actual class on the left; predicted class on top*

| | Sport | News | Politics |
| --- | --- | --- | --- |
| Sport | 5 | 3 | 0 |
| News | 2 | 3 | 1 |
| Politics | 0 | 2 | 11 |

- *easier to visualize confusion matrix via colored heatmap*
- very important: find out what positive and negative means
- $\text{Accuracy} = \frac{\text{True positive + True negative}}{\text{total observations}}$
- false positive = "false alarm"
  - eay to remember in security applications

Visualizing Classification Performance using ROC curve (receiver operating characteristic)
- positive class: malware
- negative class: benign
- ROC plots true positive rate vs false positive rate
  - TPR = % of good correctly labeled as good
  - FPR = % of good incorrectly labeled as bad
- *goal: maximize AUC (area under the curve)*
- if a machine learning algorithm achieves 0.9 AUC (out of 1), that's a great algorithm, right?
  - not necessarily
  - could be bad at detecting the extreme cases

### Quiz 2

1. The "Bayes" part of the term "Naive Bayes" comes because we use the Bayes decision rule for classification. Where does the "Naive" part come from?

> The conditional independence assumption

2. Which of the following statements does NOT hold true for the Naive Bayes classification algorithm?

> The position of the words in the document is important

3. Let A and B be two independent events with $P(A) = 0.5$ and $P(B)=0.25$. Then, using the naive assumption to the Bayes' theorem, what will be the value of $P(A,B)$?

> 0.125

4. What are the values for Precision and Recall respectively for the confusion matrix given below?

> 0.7, 0.7

5. Which of the following statements is **FALSE**?

> If a machine learning algorithm achieves 0.95 AUC score on a classification task, we can confidently say it is a very good algorithm.

