# Module 4

## Topic 1

### Lesson 1: Logistic Regression - Part 1

Generative vs Discriminative
- Generative models
  - Model prior and likelihood explicitly
  - "Generative" means able to generate synthetic data points
  - Examples: Naives Bayes, Hidden Markov Models
- Discriminative models
  - Directly estimate hte posterior probabilities
  - No need to model underlying prior and likelihood distributions
  - Examples: Logistic Regression, SVM, Neural Networks

Bayes Decision Rule


$$P(y|x) = \frac{P(x|y)P(y)}{P(x)} = \frac{P(x,y)}{\sum_y P(x,y)}$$

$$Posterior = \frac{Likelihood * Prior}{Normalization Constant}$$

- Generative: We need to calculate likelihood and prior explicitly
- Discriminative: Can we calculate Posterior directly without using Bayes equation?

Logistic Function for Posterior Probability
- Let's use the following function:
  - $P(y|x)=g(s)=\frac{e^s}{1+e^s}=\frac{1}{1+e^{-s}}$
  - $s=x\theta$
- This formula is called the sigmoid function
- It is easier to use this function for optimization
- Is 0.5 threshold cut-off a good choice?
- Many equations can give us the logit / sigmoid "S"-shape

Sigmoid Function
- $g(s) = \frac{e^s}{1+e^s} = \frac{1}{1+e^{-s}}$
- $s = \sum_{i=0}^d x_i \theta_i = \theta_0 + \theta_1x_1 + ... + \theta_d x_d$
- *soft classification posterior probability*

### Lesson 2: Logistic Regression - Part 2

Three Linear Models
1. $h(x) = sign(x\theta) \rightarrow \text{linear classification (Perceptron)}$ **hard classification**
2. $h(x) \rightarrow \text{linear regression}$
3. $h(x) = g(x\theta) \rightarrow \text{logistic regression}$ **soft classification posterior probability**

Sigmoid is interpreted as Probability
- Example: prediction of whether a customer likes a product based on the customer written feedback
- Input $x$: a BoW or TF-IDF of a document that contains a customer's feedback
- $g(s)$: a probability of whether a customer likes the product or not
  - $s=x\theta$: let's call this risk score
  - $h_{\theta}(x) = p(y|x) = \begin{cases} g(s), & y=1 \\ 1 - g(s), & y = 0\end{cases}$

Logistic Regression Model

$p(y|x) = \begin{cases}\frac{1}{1+\text{exp}(-x\theta)} & y=1 \\ 1 - \frac{1}{1+\text{exp}(-x\theta)} = \frac{\text{exp}(-x\theta)}{1+\text{exp}(-x\theta)} & y = 0\end{cases}$

We need to find $\theta$ parameters, let's set up log-likelihood for n datapoints  
$l(\theta) := \log{\Pi_{i=1}^n p(y_i |x_{i, \theta})}$  
$=\sum_i \theta^T x_i^T (y_i - 1) - \log{(1+\text{exp}(-x_i \theta))}$

The Gradient of $l(\theta)$

$l(\theta) := \log{\Pi_{i=1}^n p(y_i, | x_i, \theta)}$

$\sum_i \theta^T x_i^T (y_i - 1) - \log{(1+\text{exp}(-x_i \theta))}$

- Gradient

$\frac{\partial l(\theta)}{\partial \theta} = \sum_i x_i^T (y_i - 1) + x_i^T \frac{\text{exp}(-x_i \theta)}{1+\text{exp}(-x_i \theta)}$

- setting it to 0 does not lead to a closed form solution

Gradient Descent
- one way to solve an unconstrained optimizatino problem is gradient descent
- given an inital guess, we *iteratively* refine the guess by taking the direction of the negative gradient
- think about going down a hill by taking the steepest direction at each step
- Update rule: $x_{k+1} = x_k - \eta_k \nabla f(x_k)$
  - $\eta_k$ is called the **step size** or **learning rate**

Gradient Ascent (concave) / Descent (convex) Algorithm
- initialize parameter $\theta^0$
- do
  - $\theta^{t+1} \leftarrow \theta^t + \eta \sum_i x_i^T (y_i - 1) + x_i^T \frac{\text{exp}(-x_i \theta)}{1+\text{exp}(-x_i \theta)}$
- while the $||\theta^{t+1} - \theta^t|| > \epsilon$

Logistic Regression
- assume a threshold and...
  - predict $y = 1$ if $g(s) \geq 0.5$
  - predict $y = 0$ if $g(s) < 0.5$

Advantages and Disadvantages of Logistic Regression
- Advantages
  - Simple algorithm
  - Does not need to model prior or likelihood
  - It provides a probability output
- Disadvantages
  - We have the discriminative model assumption
  - Model needs to be optimized using a numerical approach

### Lesson 3: Support Vector Machine - Part 1

Linear Separation
- we can have different separating lines
  - why is the bigger margin better?
  - what $\theta$ maximizes the margin?
- all cases: error is zero and they are linear, so they are all good for generalization

What is the best $\theta$?
- maximum margin solution: most table under perturbations of the inputs

Finding $\theta$ that maximizes margin
- Solution (decision boundary) of the line: $x\theta = 0$
- Let $x_i$ to be the nearest data point to the line (plane)
- Decision boundary would be: $x\theta + b = 0$

What is the length of my large margin?
- $\text{distance} = \frac{1}{||\theta||}|(x_i \theta - x \theta)|$
- $\text{distance} = \frac{1}{||\theta||}|(x_i \theta + b - x\theta - b)|$
  - my constraint = $|x_i \theta + b| = 1$
  - a point on the decision line = $x\theta + b = 0$

### Lesson 4: Support Vector Machine - Part 2

Now we need to maximize the margin
- maximize $\frac{2}{||\theta||}$
- subject to min value of $|x_i \theta+b| = 1 \rightarrow nearest neighbor$ for $i = 1, 2, ..., N$
- there is a "min" in our constraint; it can be hard to optimize this problem (non-convex form)
- write the following term to get rid of the absolute value
- $|x_i \theta + b| = y_i (x_i \theta + b) \rightarrow$ for a correct classification
- if $\text{min}|x_i\theta + b| = 1 \rightarrow so it can be at least 1$

This is the same thing as...
- maximize $\frac{2}{||\theta||}$
- subject to $y_i(x_i\theta + b) \geq 1$ for $i = 1, 2, ..., N$

On the other side of the hyperplane...
- minimize $\frac{1}{2} \theta \theta^T$
- subject to $y_i(x_i\theta + b) \geq 1$ for $i = 1, 2, ..., N$

Lagrange Formulation

--------------------------

### Quiz 3

