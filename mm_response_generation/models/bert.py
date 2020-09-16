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

    def forward(self, utterances, utterances_mask, utterances_token_type):
        out_all, _ = self.bert(input_ids=utterances,
                        attention_mask=utterances_mask,
                        token_type_ids=utterances_token_type)
        return out_all
