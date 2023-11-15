import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, vocab, num_classes):
        """
        Initialize LSTM with the embedding layer, LSTM layer and a linear layer.
        
        Args:
            vocab: Vocabulary. (Refer to this for documentation: https://pytorch.org/text/stable/vocab.html)
            num_classes: Number of classes (labels).

        Returns:
            no returned value

        NOTE: Use the following variable names to initialize the parameters:
            1. self.embed_len -> the embedding dimension
            2. self.hidden_dim -> the hidden state size 
            3. self.n_layers -> the number of recurrent layers. Set the default value to 1

        HINT: Given that you're using a bi-directional LSTM, make the appropriate settings to the LSTM and Linear layers during initialization.
        """
        super(LSTM, self).__init__()
        
        raise NotImplementedError
        
        self.embed_len = 50
        self.hidden_dim = 75
        self.n_layers = 1
        self.p = 0.5 # default value of dropout rate

        self.embedding_layer = None # remove None and initialize the embedding layer
        self.lstm = None # remove None and initialize the LSTM layer
        self.linear = None # remove None and initialize the LSTM layer
        self.dropout = None # remove None and initialize the LSTM layer

    def forward(self, inputs, inputs_len):
        """
        Implement the forward function to feed the input through the model and get the output.

        Args:
            inputs : A (B, L) tensor containing the input sequences, where B = batch size and L = sequence length
            inputs_len :  A (B, ) tensor containing the lengths of the input sequences in the current batch prior to padding.

        Returns:
            output: Logits of each label. The output is a tensor of shape (B, C) where B = batch size and C = num_classes
       

        NOTE:
            1. For padding and packing sequences, consider using : torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence.
            2. Using dropout layers can also help in improving accuracy.
            3. For LSTM outputs refer to this documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html.
                (This might be helpful in correctly computing the input to the Linear layer)
        """
        raise NotImplementedError

