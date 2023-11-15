import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class SequenceLabeling(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. Bert Model using the pretrained 'bert-base-uncased' model,
            2. Dropout module.
            3. Linear layer. In dimension should be 768.

        Args:
        num_classes: Number of classes (labels).

        """
        super(SequenceLabeling, self).__init__()
        raise NotImplementedError

        self.bert = None # remove None and initialize BERT
        self.dropout = None # remove None and initialize the Dropout layer
        self.linear = None # remove None and initialize the Linear layer

    def forward(self, inputs, mask, token_type_ids):
        """
        Implement the forward function to feed the input through the bert model with inputs, mask and token type ids.
        The output of bert layer model is then fed to dropout and the linear layer. 

        Args:
            inputs: Input data. (B, L) tensor of tokens where B is batch size and L is max sequence length.
            mask: attention_mask. (B, L) tensor of binary mask.
            token_type_ids: token type ids. (B, L) tensor

        Returns:
            output: Logits of each label. (B, L, C) tensor of logits where C is number of classes.
        """
        raise NotImplementedError