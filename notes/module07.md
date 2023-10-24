# Module 7

## Topic 1

### Lesson 1: Deep Learning CNN

Deep Learning
- deep neural networks learn hierarchical feature representations

Smaller Network: ANN
- we know it is good to learn a small model
- from this fully connected model, do we really need all the edges?
- can some of these be shared?
- when it comes to images, some patterns are much smaller than the whole image
  - can represent a small region (like the beak of a bird) with fewer parameters

Why CNN works well for images
- we can train a lot of "small" detectors and each detector must "move around"?

Can we use CNN for NLP?
- yes, but we do need to prepare a dataset which is readable to a CNN model like an image which is a matrix

How to prepare the dataset for an NLP problem?
- Let's say we have 3 documents collected from emails:
  - Wafa and Mahdi teach NLP class
  - NLP is neat
  - CNN is a good model
- Our vocabulary vector has 12 unique or distinct words; i.e. wafa, and, madhi, ...
- The longest document has 6 words (NOTE that CNN needs all datapoints to have the same size)

Applying One-Hot Encoding
- wafa = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- and = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- mahdi = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- ...
- each unique word will be a vector with a length equal to 12
- there are 12 unique words in our vocabulary vector

Convering Documents into a Matrix
- stack all of the one-hot encoding vectors till you have an n x d matrix where n is the number of words in the **longest** document and d is the number of unique words in the vocabulary
- all documents have the same size as the longest document in the corpus
- this is achieved by zero padding
- all documents can be converted the same way

### Lesson 2: Deep Learning CNN

Important Concept: A Convolutional Layer
- Now that all documents have the same size and are converted into a matrix form; we can start feeding them into a CNN model
- A CNN is a neural network with some convolutional layers (and some other layers)
- A convolutional layer has several filters that do the convolutional operation

Convolution vs Fully Connected
- CNN has fewer parameters!
- Shared weights

Max Pooling
- each filter is a channel
- the number of channels is the number of filters

Example
- original: 6x6
- convolutional layer: 4x4
  - feature engineering approach: creates richer input containing more information
- max pooling: 2x2
- flatten convoluted data and feed into fully connected feed forward network

Another Example:
- orig:  1 x 24 x 20
- conv: 25 x 22 x 18
- maxp: 25 x 11 x 9
- conv: 50 x  9 x  7
- maxp: 50 x  4 x  3
- flattened: 600

A CNN Compresses a Fully Connected Network in 3 Ways:
1. Reducing the number of connections
2. Shared weights on the edges
3. Max pooling further reduces the complexity

Example:
- `input_shape = (24, 20, 1)`: 24 x 20 matrix, 1 channel
- `Model2.add(Convolution2D(25, 3, 3))`: there are 25 3x3 filters

## Topic 2

### Lesson 1: Deep Learning RNN

Named Entity Recognition (NER) using RNN
- Example document: Mahdi and Wafa teach NLP
- Our simple NER model will detect whether each word is referring to a person or not. In our example, 1 refers to a person and 0, not a person
  | Mahdi | and | Wafa | teach | NLP |
  | --- | --- | --- | --- | --- |
  | 1 | 0 | 1 | 0 | 0 |
- For NER, the location of each word in a sentence is very important

Encode Each Word into a Vector
- we can use an encoding method to convert words into a vector (Word2Vec, GloVe, One-hot encoding, ...)
- Mahdi and Wafa Teach NLP >> each word $(x)$ will be a $d$ dimensional vector: $x \in \mathbb{R}^d$
- Output or prediction will be a neuron $O \in \mathbb{R}^m$
- the size of $y$ depends on our desired output
- for our NER problem, $y$ will be a scalar
- model parameters or weights $\theta_{dxm}$ ($d$: size of each vectorized word and $m$: number of hidden neurons)
- for each neuron ($x_i$), we have $m$ parameters

### Lesson 2: Deep Learning RNN

Let's go back to our NER problem using a Feed-Forward Approach
- $m$: number of hidden neurons is a **hyper-parameter** and needs to be optimized
- there are two sets of different parameters
  - (1) input to hidden layers: $\theta^1_{dxm}$
  - (2) hidden layers to output: $\theta^2_{mxy}$

RNN Concept to Connect Different Time Steps
- We introduced a new set of parameters ($\theta_{mxm}^3$) which generates a new vector of hidden neurons ($h_t$) with size $m$

Simple Representation of RNN
- rnn_units
- `tf.keras.layers.SimpleRNN(rnn_units)`

Forward Pass: How to Calculate Past Memory (h) in RNN

![rnn unit](./../lectures/module07/rnn_unit.png)

- $\theta^{(1)}$: weight (parameter) matrix associated with input data
- $\theta^{(2)}$: weight (parameter) matrix associated with output data
- $\theta^{(3)}$: weight (parameter) matrix associated with hidden state
- $h_t = f(x_t, h_{t-1}, \theta)$
  - $f$: activation function such as $tanh$ (hyperbolic tangent)
  - $x_t$: input
  - $h_{t-1}$: past memory (previous step)
  - $\theta$: model parameters (weight)
$h_t = \text{tanh}(x_t \theta^{(1)} + h_{t-1} \theta^{(3)} + b)$
    - each term will be a vector of size $m$
    - $b$ representes the bias vector here

Forward Pass: How to Calculate Output for Each Step ($\hat y$) in RNN:
- $\hat{y_t} = \text{softmax}(h_t \theta^{(2)})$
  - softmax scales the output between 0 and 1
  - the output of $h_t\theta^{(2)}$ will be a scalar for our NER problem

The complementary step to forward pass is backpropagation (Backproppagation through time (BPTT))

Different RNN Models
- **One-to-Many**: text generation; image captioning
- **Many-to-One**: sentiment classification
- **Many-to-Many**: part of speech tagging (POS), NER, translation, forecasting

Some Problems with RNN
- forward pass, backpropagation and repeated gradient computation can lead to two major issues
- **Exploding gradient** (high gradient values leading to very different weights in every optimization iteration)
  - Solution: gradient clipping (clip a gradient when it goes higher than a threshold)
- **Vanishing gradient** (low gradient values that stall the model from optimizing the parameters)
  - Solution: ReLu activation function
  - LSTM, GRUs (different architectures)
  - better weight initialization
- RNN suffers from short-term memory for a long sentence where words of interest may be palced far frome ach other in a sentence (vanishing gradient). Other architectures such as LSTM and GRUs can help with this

--------------------------------------

### Topic 1 Lesson 1

Hi class. In this lecture, we will go over a deep learning method which is called a convolutional neural network, commonly known as CNN. As the name implies, some convolution tasks are involved in creating the network. Cnn is a superior method. Compare it to artificial neural network for the type of dataset where position matters, such as photos and potentially documents. We will go over what a deep learning model is and how a convolutional neural network could help us improve the model. In addition, we will go over the convolutional task and how it reduces the number of parameters which allows us to reduce the danger of overfitting. Deep learning is a network that contains several hidden layers. In a deep-learning and network without any conclusion part, input data, such as images, are fed into a network in a raw format, e.g. each pixel is considered as a feature. Then these features create the input neurons. And if we use FC or fully connected layers to create hidden layers, it will lead to a very large number of parameters, which makes the model more complex and not generalizable to our test data. Therefore, the danger of overfitting. In the example provided on this slide, there are three hidden layers. And the first hidden layer is created based on the linear combination of the input neurons pass into an activation function. Typically, these initial layers identify light and dark pixels, edges and simple shapes in images. The second layer is constructed based on the linear combination of the first hidden layer, which is a linear combination of our input neurons. This helps the model learn a more complex structure. The middle hidden layers are generally responsible for more complex shapes and objects. The final layers are responsible for detecting the main objective or the goal of the network. In our example shown on this slide, it would be the human face. Smaller artificial neural networks are good when we want to learn a smaller models are our data points are not very complex, or we don't deal with big data. But the question is, do we really need all the edges? Can we reduce the number of parameters? Can some of these parameters be shared among different features or new routes? For the sake of simplicity, let's consider learning an image. And our main goal is to construct a network that can lend a bird's beaks. Essentially, our model would be a type of beak detector in a converged, in the convolutional fully connected layer, we consider each pixel as a neuron. That means we remove the regional and local dependency among the features when you feed them into a network. As every Then in our big detector network, we need to be aware of the surrounding features of pixels to simply an, accurately realize whether an image has a peak or not. Primarily, looking at one pixel at a time requires a network to have many parameters to initially find the local relationship among pixels and then figure out whether there is a beak in an image or not. Now the question becomes, can we help them network, but providing the local relationship in advance? And this will be answered in the upcoming slides. If you have also notice, I use a word called simple, which means using fewer parameters to come up with a beak detector, making our model simpler. Now, if we were to introduce a small regions to a network instead of pixel by pixel, how our network is going to learn that these small region or big detectors are commonly known as kernels, are filters. Logically, we cannot have one filter to tag any type of bird's beak. As an example, we have sparrows and dark here. Yes, there are both birds have peaks, but they are different shapes. Therefore, we need to introduce several filters that detect different type or shades of birds beaks. These filtered will be our model parameters that need to be learned. I have been talking about CNN through image examples, but our class is about documents and tax. And the question becomes, can we really use CNN for documents? The short answer is yes. But we need to prepare our data and create a matrix from each document that is compatible with a convolutional neural network model. For simplicity, let's say we have three documents, three sample data points. These documents are collected from e-mails. First document is raffle and Maddie teach NLP class. The second one, NLP is neat and the last one, CNN, is a good model. Our vocabulary vector has 12 unique or distinct words, e.g. Rafa and mathy and the server, etc. The longest document has six wars. This is very important to know that CNN requires all the data points to have the same size. However, in our example, each sentence has a different number of words. Can we make him have the same size? We will see that. First thing first, we need to use an encoding technique to convert every single word into a vector. Because our vocabulary has a size equal to tool, each word vector needs to be 12th. So old lectures, awards will have the same length. We employed a one-hot encoding approach to encode words into vectors. We can use more advanced techniques to do that, such as the Word2vec algorithm. Now we need to convert the document or sentence into a matrix. Our first document has six words, which happens to be the longest document in our example. Once we put them together, we create a matrix with the size of six one-twelfth. The rows indicate the number of words and columns created based on our word vector encoding technique. Now let's look at the second document. Wait a second. I believe we say that CNN, these are all input data to have the same size. But this example curie is a different size compared to the previous example. A good trick to deal with this issue is to pad the current matrix by zeros to have the same size as our longest document. Therefore, all documents will have the same size as the longest document in the corpus. And this is achieved by zero-padding. We follow the same techniques for the rest of the documents.


### Topic 1 Lesson 2

Hi class. In this lecture, we will continue our CNN by going over the convolutional and prediction part of this deep learning model. Now that all documents have the same size and are converted into a matrix form, we can start feeding them into a CNN model. A CNN is a neural network with some convolutional layers and some other layers which we will cover. A curve evolution or layer has several filters that do the convolution operation. Anytime we talk about conclusion, it means that these filters need to slide over the input matrix and do a dot product operation. For simplicity, let's say we have input data that is six by six. If we want to use a ANN approach with fully connected layers, please look at the example shown on the bottom of the slide. We need to consider each element of the matrix as a feature or neuron. Let's say our first hidden layers have the same number of neurons as our inputs. Then the number of edges between the input layer and the first hidden layer will create 36 multiplied by 36 parameters that can increase the chance of overfeeding by making the model more complex. But looking at the example shown on the top of the slides, if we were to use a CNN approach, we needed 18 parameters. If we use two filters where each of them has a size of three by three. We know that in R-CNN model, we need to learn the filters elements. And typically a three-by-three or five-by-five filter is used in a CNN model. The number of filters is a hyperparameter. In a CNN model, all the fields are, values are randomly initialized at fairs and then they will be optimized in the backpropagation process, similar to an artificial neural network or CNN. Let's look at our six-by-six image or input data beat. Let's look at our six by six input data and understand how the convolutional part is done and wide leads to reducing the number of parameters. Filter one is to slide over the original input data. The filter is placed on top, on top left of the input matrix. And the sum product operation is done between the filter and the sub matrix portion of our input data which intersects with the filter. The sum product will lead to a scalar number, which is three in this example, for the first slide, if we vectorize or flatten all the input elements shown on the right side of this slide. The new neuron in the hidden layer is created by connecting it to only nine neurons of the input instead of all 36. This process will help significantly reduced the number of parameters. In addition to using only nine neurons from the input to generate a new neuron in the first hidden layer. Cnn takes advantage of parameter sharing. To do this, we need to slide over the filter by one to the right of the original input matrix. When we slide the filter by one element at a time, it uses a stride one. The next new neuron in the hidden layer is created by moving the filter with one stride to the right and calculating the sum product value, the new value will be as negative one. It is very important to note that the values of filter one do not change and it stays the same. In fact, we use the same values of filter, one for each convolution operation to generate a new neuron. This is commonly known as sharing the weights among different neurons. Let's flashback quickly to our big detector. The big detector for a sparrow should not be changed when it's slid over an image. This is a whole CNN architecture and so forth. What we have done, we took care of the document input and the convolutional part. Now we need to learn what is the max pooling which is commonly used in CNN. When we convolute our input matrix by a filter, it will produce a new matrix smaller than original one. E.g. for a six-by-six input matrix, if we convolute that with a three-by-three filter, the output matrix or the new feature map will be four by four, which is a smaller than the original input data. To further reduce the model's parameter, max pooling is typically performed after the convolutional tasks, which acts as a sub-sampling or resizing after feature map. A two-by-two block max-pooling is commonly used in CNN architectures. As an example, if we apply a two-by-two max pooling over a matrix of size of four by four, it will reduce its size by half. And the new matrix will be two-by-two. Essentially, max pooling will replace a 2-by-2 sub-matrix with a scalar with the highest value among the 2-by-2 elements. If you use two filters and perform the operation of convolution and max polling on an input matrix with the size of six by six. The outputs will be a two-by-two matrix with two channels. Each channel is generated by a filter matrix. Therefore, a CNN compresses a fully connected network in three ways. First, reducing number of connections. Second, shared weights on the edges. And the third, max-pooling further reduces the complexity. After the task of max-pooling, essentially, we create a new data or feature map, and we can do the same process for another convolutional layer accompanied by a max pooling. The choice of the number of convolutional and max pooling layers depends on the model architecture. And one can come up with any architecture by needs to be optimized to perform as accurately as possible without overfitting. I often considered the convolutional parts of a CNN model as a feature engineering approach. In a convenient convolution artificial neural network, we feed the raw information of the input data into a neural network model. While in CNN, we preprocess those inputs using convolutional layers to create a richer input that contains more information. And then we feed those as an input of a neural network model. As you can see on this slide, after we're done with the convolutional part of CNN, we need to flatten the matrix. This is slide shows a flattened version of convoluted data which are fed into a fully connected neural network. The last layer of the neural network neural network model defines the objective of the network, e.g. it could be a sentiment analysis of documents similar to a neural network in order to optimize the parameters of the fully connected neural networks and the filters of a convolutional layers, we need to use a backpropagation approach similar to an ANN model. All these tasks can be done using some well-known libraries such as Keras or Pi torch, e.g. in Keras, we first initialize the model and we define the first convolution with this number of filters. Here in this example, we use in 25 filters where each filter size is three by three. And we also need to define the input size, where we define it to have one channel and its size is 24 by 20. Once we added a convolutional layer, we add a two-by-two max pooling on top of that, which reduce the size of the feature is mapped by two. This example shows the evolution of matrix size by adding different convolutional layers and max-pooling. Note that the input matrix is one by 24 by 20 and its size becomes 25 by 11 by nine after applying 25 filters on top of the input matrix. And the max pooling operation. In the second convolutional layer, we use 50 filters and the input of the new convolutional layer is the output of the previous convolutional layer. Because the input data after the first convolutional layer has 25 channels, the filter is must have 25, 25 channels too. E.g. each field there must be 25 by three by three. Once we're done with the convolutional layer, we flatten the matrix and add it to our fully connected neural networks. Here we choose to have two fc layers or fully connected layers. Into CNN lecture, we'll learn about deep neural network. We went over the sealant model and convolutional part. We converted the corpus into a matrix which can be fed into a CNN model. We use Keras library to create a CNN architecture.

### Topic 2 Lesson 1

Hi everyone. In this class, we will go over a very well-known deep learning model, which is called recurrent neural network or RNN. This type of network is essentially the more applicable in natural language processing, where the position of words can affect the final result. Using RNN, we're going to step into a temporal dimension using a neural network to handle sequential data. A convolutional neural network does consider the local region of the input data, but it does not have a notion of time or sequence. RNN models were created because there were a few issues in the feed-forward neural network, such as convolutional neural networks, for example, these models cannot handle sequential data, considers only the current input, cannot memorize previous inputs. The solution to these issues is a recurrent neural network or RNN. And RNN can handle sequential data, accepting the current input data and previously receive inputs. RNNs can memorize previous inputs due to their internal memory. Let's expand the concept of internal memory of RNN using the name entity recognition example or NER. Our document example is Mahdi and Wafa teach NLP. Our simpler NER model, we detect whether each word is referring to a person or not. In our example, one refers to a person and zero another person. For any of the location of each word in a sentence is very important to detect whether it is a person or not. For example, verb cannot be a person and its location is right next to or very close to the subject. Therefore, our model needs to take the location of each word into the consideration. Before going into the details of an RNN model, we need to prepare our data and do some pre-processing on each ward, by converting them into a vector. We can use an encoding method to convert words into a vector such as Word2vec, Glove, One-hot encoding, and etc. It is very important to pay close attention to the size of vectors and matrices from this slide on. And also the math notation I used to explain them, for example, I use the math notation of x for each word, and x is a vector of length d. This means that every word in our vocabulary is a vector of length d. This slide shows an example of a feedforward neural networks such as ANN and CNN. Based on our previous slide, each word has the elements, therefore, the input will have the neurons, commonly, we add an additional neuron to take care of the bias term. In this example, our hidden layer has two learning blocks that produce two new neurons. Because we're using the FC or fully connected layer, we need to linearly combine all the input layers for each learning block and feed its input into an activation function. A word is denoted by x and I use x sub i as a generic index where i goes from 1-d. This type of network visualization shows a very detailed network operation for each new neuron. We can further compress this type of visualization to extend it to networks such as RNN. Here I tried to simplify the visualization of the network shown on the left side. For now, let's just focus on the left side of this slide. Each word is a vector of length d, leading to d neurons for our input layer, these neurons are connected to learning blocks which generate new neurons or hidden neurons. I use lowercase m as the math notation for the number of learning blocks that generate m new hidden neurons. Because we use a fully-connected layer, each input layer is connected to all m learning blocks that needs m parameters. The parameters are shown with the math notation of Theta here, so each input neuron is m parameters. Therefore, the number of parameters between the input layer and the hidden layer is denoted by a matrix named Theta with the size of d by m. If you want to just have one hidden layer in this network, the hidden neurons which are defined as all will be connected to the final output learning block. In any our example, our task is to predict whether a word refers to a person or not, therefore, the output of the neural network could be just one sigmoid learning block. Therefore, the output, which is denoted as y, will be a scalar value between zero and one in our example. Now let's focus on the right side of this slide, which shows a very compact visualization of the same network shown on the left side. The math notation of a word is represented by x sub t, t here refers to a sequential location of reward. In other problems such as stock prediction, t will refer to a temporal notion of the input data. X sub t, or a word requires d by m parameters. The hidden neurons will be connected to output learning block. The number of parameters between the hidden neurons and the output neuron is defined as n by y and is determined based on the type of classification, whether it is binary or multi-label. In our binary NER classification, it will be just a single learning block. Note that we need to name entity recognition for every single word and the output prediction for each word is represented by y hat sub t. [MUSIC]

### Topic 2 Lesson 2

Hi everyone. In this lecture we will continue our RNN class and explain how to create the RNN structure by adding the feed-forward network in a sequential form and finally perform the name entity recognition task. In our previous lecture, for each board, we created a feed-forward network and represented it using a compact visualization. Our example problem was the NER task for Mahdi and Wafa teach NLP. In this example, we have five words, which means that we need to construct five similar feed-forward network for each word. In our previous lecture, we showed the compact visualization in a horizontal view. Here we turn it into a vertical view so we can visualize it for all the feed-forward units, for all the words in our document. M is the number of hidden neurons and is considered a hyperparameter that it needs to be optimized. If you have two sets of different parameters, one input to hidden layer shown as Theta, with superscript one and subscript d by m. Two, hidden layers to output as Theta, with superscript two and subscript n by y. It is interesting to note that all these feed-forward network units are isolated and they're not connected to each other. We know that the location of each word matters for NER problem. And therefore, we need to find a way to connect all these units together in a sequential way. In order to communicate between our first word feed-forward unit and the second word, we connect them through hidden neurons shown as h. Adding a new set of communications between feed-forward units will impose a set of new parameters. The communication parameters will have a size of m by m, and the hidden neurons shown that h will be a vector with a length of m. We will soon go over there how the neurons in the hidden states are calculated and how they can be concatenated with a new word for the second feed-forward network, for example, h_0, which is generated by the first feed-forward unit will be concatenated by the word and where they produce new communication hidden neurons, which are named as h_1. It is important to pay attention to the superscripts of the parameters in our RNN model. We have three superscripts of 1, 2, and 3. Parameters sharing plays a major role in RNN models. This means that all Thetas with superscript one, which are the parameters associated with the input data will be shared throughout the network. The same goes for the parameters with superscript two and three. Hence, the total number of parameters learned by this model will be d by m plus m by y plus m by m, regardless of the number of words in a sentence. In addition, each document may contain a different number of words and it will not be efficient to create an RNN model for each document. In practice, zero-padding is used to construct one RNN model, which covers all the input documents with the different sizes. If the difference between a document that has the maximum number of words and a document that has the minimum number of words is high, a budgeting approach is taken into account. This means that we pad the sequence with different bucket lens of word lengths, for example, 5, 20, 50, 100 documents buckets, and then we create an RNN model for each bucket. And this is like a simple representation of an RNN model with this recurring state is shown. In TensorFlow Keras, the simple RNN layers can be easily constructed by calling the simple RNN method. On the left side of this slide, I provided a compressed visualization of an RNN unit. Theta with superscript one represents weights or parameters matrix associated with the input data. Theta with superscript two represents weights or parameters matrices associated with the output data. Theta with superscript three represents weights or parameters matrix associated with hidden state. Now we need to understand how we calculate the hidden neurons h, which take care of the memorization part of RNN. In fact, the hidden state helps the network memorize the previous step. In order to calculate the hidden state h_t, using the previous step, h_t minus 1, we need to combine them together using the hyperbolic tangent shown on this slide. In each step, the output neuron is calculated using a softmax or sigmoid in a special case of a binary classification. Note that h_t is calculated using the current word vector and the hidden state neurons of the previous step. To call this four arrows are shown here. The blue arrows refer to the forward pass operation, and the orange one refers to the backpropagation process to optimize the model parameters. In each RNN unit, we need to calculate the loss function locally. In this example, we have five words and we will have five loss function for each unit. And the summation of all these loss function will provide us the total loss. In the backpropagation part, we move backward for each local loss to optimize the parameters. And anytime the backpropagation path of two local loss function intersects with each other, they need to be summed up. There are different RNN models depending on what type of problem you are going to solve. In RNN name entity recognition, we use a many-to-many architecture. Because for each RNN unit, we need to know the output of whether a word refers to a person or not. One-to-many is used for text generation or image captioning, for example, input is one image and we need to generate tags for that. We also have many-to-one, which is commonly used for sentiment analysis. We have a document with multiple words as input, and we want to detect whether a document as a whole is a positive or negative sentence through a sentiment analysis. Forward pass, backpropagation and repeated gradient computation can lead to two major issues. The first one is exploding gradient, where high gradient values lead to very different weights in every optimization iteration. A possible solution to avoid this issue is to use gradient clipping, meaning clip a gradient when it goes higher than a threshold. The second one is a vanishing gradient where low gradient values will stall the model from optimizing the parameters. There are some ways to cue the issue, for example, by using a ReLu activation function, other architectures such as LSTM or GRUs. And the last one is to utilize a better weight initialization. In this lecture, we learned about RNN and how it memorizes prior information. I talked about many-to-many RNN architectures to an NER example, also briefly discuss some potential issues and solutions to those issues in an RNN model.
