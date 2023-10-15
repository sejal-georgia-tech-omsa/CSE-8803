import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab, num_classes):
        '''
        Initialize RNN with the embedding layer, bidirectional RNN layer and a linear layer with a dropout.
    
        Args:
        vocab: Vocabulary.
        num_classes: Number of classes (labels).

        NOTE: Please name the layers self.embedding, self.rnn and self.linear to ensure the local tests run properly.
        
        '''
        super(RNN, self).__init__()
        self.embed_len = 50  # embedding_dim default value for embedding layer
        self.hidden_dim = 75 # hidden_dim default value for rnn layer
        self.n_layers = 1    # num_layers default value for rnn
        self.p = 0.5   # default value for the dropout probability, you may change this

        raise NotImplementedError

        self.embedding = # initialize embedding layer
        self.rnn = # initialize RNN layer
        self.linear = # initialize linear layer

    def forward(self, inputs, inputs_len):
        '''
        Implement the forward function to feed the input through the model and get the output.

        You can implement the forward pass of this model by following the steps below. We have broken them up into 3 additional 
        methods to allow students to easily test and debug their implementation with the help of the local tests.

        1. Pass the input sequences through the embedding layer to obtain the embeddings. This step should be implemented in forward_embed().
        2. Pass the embeddings through the rnn layer to obtain the output. This step should be implemented in forward_rnn().
        3. Concatenate the hidden states of the rnn as shown in the architecture diagram in HW3.ipynb. This step should be implemented in forward_concat().
        4. Pass the output from step 3 through the linear layer.

        USEFUL TIP: Using dropout layers can also help in improving accuracy.

        Args:
            inputs : A (B, L) tensor containing the input sequences, where B = batch size and L = sequence length
            inputs_len :  A (B, ) tensor containing the lengths of the input sequences in the current batch prior to padding.

        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes
        '''
        raise NotImplementedError

    def forward_embed(self, inputs):
        """
        Pass the input sequences through the embedding layer.

        Args: 
            inputs : A (B, L) tensor containing the input sequences

        Returns: 
            embeddings : A (B, L, E) tensor containing the embeddings corresponding to the input sequences, where E = embedding length.
        """
        raise NotImplementedError
    
    def forward_rnn(self, embeddings, inputs_len):
        """
        Pack the input sequence embeddings, and then pass it through the RNN layer to get the output from the RNN layer, which should be padded.

        Args: 
            embeddings : A (B, L, E) tensor containing the embeddings corresponding to the input sequences.
            inputs_len : A (B, ) tensor containing the lengths of the input sequences prior to padding.

        Returns: 
            output : A (B, L', 2 * H) tensor containing the output of the RNN. L' = the max sequence length in the batch (prior to padding) = max(inputs_len), and H = the hidden embedding size.
        
        HINT: For packing and padding sequences, consider using : torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence. Set 'batch_first' = True and enforce_sorted = False (for packing)
        """
        raise NotImplementedError
    
    def forward_concat(self, rnn_output, inputs_len):
        """
        Concatenate the first hidden state in the reverse direction and the last hidden state in the forward direction of the bidirectional RNN. 
        Take a look at the architecture diagram of our model in HW3.ipynb to visually see how this is done.

        Args: 
            rnn_output : A (B, L', 2 * H) tensor containing the output of the RNN.
            inputs_len : A (B, ) tensor containing the lengths of the input sequences prior to padding.

        Returns: 
            concat : A (B, 2 * H) tensor containing the two hidden states concatenated together.
        
        HINT: Refer to https://pytorch.org/docs/stable/generated/torch.nn.RNN.html to see what the output of the RNN looks like. 
        """
        raise NotImplementedError