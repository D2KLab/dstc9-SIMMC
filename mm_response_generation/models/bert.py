import pdb

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class BertEncoder(nn.Module):

    def __init__(self, pretrained, freeze=False):
        super(BertEncoder, self).__init__()
        configuration = BertConfig()
        self.bert = BertModel(config=configuration).from_pretrained(pretrained)
        self.configuration = self.bert.config
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input, input_mask, input_token_type):
        out_all, _ = self.bert(input_ids=input,
                        attention_mask=input_mask,
                        token_type_ids=input_token_type)
        return out_all
