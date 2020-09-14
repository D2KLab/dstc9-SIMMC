from transformers import BertConfig, BertModel
import torch
import torch.nn as nn

class BertEncoder(nn.Module):

    def __init__(self):
        super(BertEncoder).__init__()
        configuration = BertConfig()
        self.bert = BertModel(config=configuration).from_pretrained('bert-base-uncased')
        self.bert.from_pretrained()
        self.configuration = self.bert.config

    def forward(self, utterances, history, item):
        pass


