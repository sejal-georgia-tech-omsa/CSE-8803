import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(1,2,3,5)):
        '''
        Initialize CNN with the embedding layer, convolutional layer and a linear layer. 
        Steps to initialize -
        1. For embedding layer, create embedding from pretrained model using w2vmodel weights vectors, and padding.
        nn.Embedding would be useful.

        2. Create convolutional layers with the given window_sizes and padding of (window_size - 1, 0).
        nn.Conv2d would be useful. Additionally, nn.ModuleList might also be useful.

        3. Linear layer with num_classes output.

        Args:
            w2vmodel: Pre-trained word2vec model.
            num_classes: Number of classes (labels).
            window_sizes: Window size for the convolution kernel.
        '''
        super(CNN, self).__init__()
        weights = w2vmodel.wv # use this to initialize the embedding layer
        EMBEDDING_SIZE = 500  # Note that this is the embedding dimension in the word2vec model passed in as w2vmodel.wv
        NUM_FILTERS = 10      # Number of filters in CNN

        # initialize embedding layer
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), freeze=False)

        # initialize convolution layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=NUM_FILTERS, kernel_size=(window_size, EMBEDDING_SIZE), padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        # initialize linear layer
        self.fc = nn.Linear(in_features=NUM_FILTERS * len(window_sizes), out_features=num_classes)

    def forward(self, x):
        '''
        Implement the forward function to feed the input through the model and get the output.

        You can implement the forward pass of this model by following the steps below. We have broken them up into 2 additional 
        methods to allow students to easily test and debug their implementation with the help of the local tests.

        1. Feed the input through the embedding layer. This step should be implemented in forward_embed().
        2. Feed the result through the convolution layers. This step should be implemented in forward_convs().
        3. Feed the output from convolution through a fully connected layer to get the logits. You may need to manipulate
           the convolution output in order to feed it into a linear layer. 

        Args:
            inputs: Input data. A tensor of size (B, T) containing the input sequences, where B = batch size and T = sequence length

        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes
        '''
        

        # STEP 1: Feed the input through the embedding layer
        embedded = self.forward_embed(x)
        
        # STEP 2: feed the result through the convolution layers
        convolution_output = self.forward_convs(embedded).flatten(start_dim=1)
        
        # STEP 3: feed the output from convolution through an FC layer to get the logits
        logits = self.fc(convolution_output)
        
        return logits

    def forward_embed(self, x):
        '''
        Pass the input through the embedding layer.

        Args: 
            inputs : A tensor of shape (B, T) containing the input sequences 

        Returns: 
            embeddings : A (B, T, E) tensor containing the embeddings corresponding to the input sequences, where E = embedding size.

        '''
        embeddings = self.embedding(x)
        return embeddings

    def forward_convs(self, embed):
        '''
        Pass the result of the embedding function through the convolution layers. For convolution layers, pass the convolution output through
        tanh and max_pool1d. 
        Args:
            input: A tensor of size (B, T, E) containing the embeddings corresponding to the original input sequences.
        Returns:
            output: A tensor of size (B, F, K) where F = number of filters and K = len(window_sizes)

        NOTE: Modify the output of the embedding layer accordingly for the convolutions.
        You may need to use pytorch's squeeze and unsqueeze function to reshape some of the tensors before and after convolving.
        '''
        
         # Reshape the embedding tensor to (B, 1, T, E)
        embedded = embed.unsqueeze(1)

        # Feed embedding tensor through convolutional layer using tanh
        conved_embeddings = [F.tanh(conv(embedded)).squeeze(3) for conv in self.convs]
        
        # Feed embedding tensor through convolutional layer using max_pool1d
        max_pooled = [F.max_pool1d(conv, conv.size(2)) for conv in conved_embeddings]

        # Concatenate the outputs of the convolutional layers
        output = torch.cat(max_pooled, dim=2)

        return output

