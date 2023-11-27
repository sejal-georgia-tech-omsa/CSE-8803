import torch

torch.manual_seed(10)
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the following modules:
            1. Bert Model using the pretrained 'bert-base-uncased' model (use from_pretrained method), Ref: https://huggingface.co/transformers/v3.0.2/model_doc/bert.html
            2. Linear layer. In dimension should be 768.
            3. Dropout module.

        Args:
            num_classes: Number of classes (labels).

        """
        super(BERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, num_classes)

    def forward(self, inputs, mask):
        """
        Implement the forward function to feed the input through the bert model with inputs and mask.
        The output of bert layer model is then fed to dropout and the linear layer. 

        Args:
            inputs: Input data. (B, L) tensor of tokens where B is batch size and L is max sequence length.
            mask: attention_mask. (B, L) tensor of binary mask.

        Returns:
            output: Logits of each label. (B, C) tensor of logits where C is number of classes.
        """
        outputs = self.bert(inputs, attention_mask=mask)  # (B, L, H)
        pooled_output = outputs[1]  # (B, H)
        output = self.dropout(pooled_output)
        output = self.linear(output)  # (B, num_classes)
        return output

