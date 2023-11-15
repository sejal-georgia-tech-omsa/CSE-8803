import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    def __init__(self, vocab, num_classes):
        """
        Initialize the LSTM + Attention model with the embedding layer, bi-LSTM layer and a linear layer.
        
        Args:
            vocab: Vocabulary. (Refer to the documentation as specified in lstm.py)
            num_classes: Number of classes (labels).
        
        Returns:
            no returned value

        NOTE: Use the following variable names to initialize the parameters:
            1. self.embed_len -> the embedding dimension
            2. self.hidden_dim -> the hidden state size 
            3. self.n_layers -> the number of recurrent layers. Set the default value to 1
            
        NOTE: 1. A context layer also needs to be defined for attention. 
              2. Adding a dropout layer would be useful
        """
        super(Attention, self).__init__()
        
        raise NotImplementedError

        self.embed_len = 50
        self.hidden_dim = 75
        self.n_layers = 1
        self.p = 0.5

        self.embedding_layer = None # remove None and initialize the embedding layer
        self.lstm = None # remove None and initialize the LSTM layer
        self.linear = None # remove None and initialize the Linear layer
        self.dropout = None # remove None and initialize the Dropout layer
        self.context_layer = None # remove None and initialize the Context layer

    def forward(self, inputs, inputs_len):
        """
        Implement the forward function to feed the input through the model and get the output.

        You can implement the forward pass of this model by following the steps below. We have broken them up into 3 additional 
        methods to allow students to easily test and debug their implementation with the help of the local tests.
        1. Pass the input sequences through the embedding layer and lstm layer to obtain the lstm output and lstm final hidden state. This step should be implemented in forward_lstm().
        2. Compute the normalized attention weights from the lstm output and final hidden state. This step should be implemented in forward_attention().
        3. Compute the context vector, concatenate with the final hidden state and pass it through the context layer. This step should be implemented in forward_context().
        4. Pass the output from step 3 through the linear layer.

        USEFUL TIP: Using dropout layers can also help in improving accuracy.

        Args:
            inputs : A (B, L) tensor containing the input sequences, where B = batch size and L = sequence length
            inputs_len :  A (B, ) tensor containing the lengths of the input sequences in the current batch prior to padding.

        Returns:
            output: Logits of each label. A tensor of size (B, C) where B = batch size and C = num_classes
        """
        raise NotImplementedError
        

    
    def forward_lstm(self, inputs, inputs_len):
        """
        Pass the input sequences through the embedding layer and the LSTM layer to obtain the LSTM output and final hidden state.
        Concatenate the final forward and reverse hidden states before returning.

        Args: 
            inputs : A (N, L) tensor containing the input sequences
            inputs_len : A (N, ) tensor containing the lengths of the input sequences prior to padding.

        Returns: 
            output : A (N, L', 2 * H) tensor containing the output of the LSTM. L' = the max sequence length in the batch (prior to padding) = max(inputs_len), and H = the hidden embedding size.
            hidden_concat : A (N, 2 * H) tensor containing the forward and reverse hidden states concetenated along the last dimension.
        
        HINT: For packing and padding sequences, consider using : torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence. Set 'batch_first' = True and enforce_sorted = False (for packing)
        
        """
        raise NotImplementedError

    
    def forward_attention(self, lstm_output, hidden_concat):
        """
        Compute the unnormalized attention weights using the outputs of forward_lstm(). You may find torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html) helpful.
        Then, compute the normalized attention weights with the help of a softmax operation.

        Args:
            lstm_output : A (N, L', 2 * H) tensor containing the output of the LSTM.
            hidden_concat : A (N, 2 * H) tensor containing the forward and reverse hidden states

        Returns:
            attention_weights : A (N, L') tensor containing the normalized attention weights.
        
        """
        raise NotImplementedError
    
    def forward_context(self, lstm_output, attn_weights, hidden_concat):
        """
        Compute the context, which is the weighted sum of the lstm output (the coefficients are the attention weights). Then, concatenate
        the context with the hidden state, and pass it through the context layer + tanh().

        Args:
            lstm_output : A (N, L', 2 * H) tensor containing the output of the LSTM.
            attn_weights : A (N, L') tensor containing the normalized attention weights.
            hidden_concat : A (N, 2 * H) tensor containing the forward and reverse hidden states

        Return:
            context_output : A (N, 2 * H) tensor containing the output of the context layer.

        HINT: torch.bmm may be helpful in computing the context.
        """
        raise NotImplementedError













