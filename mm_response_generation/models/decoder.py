import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self,
                d_model,
                d_enc,
                d_k,
                d_v,
                d_f,
                n_layers,
                n_heads,
                embedding_net,
                vocab_size,
                dropout_prob,
                start_id,
                end_id):
        super(Decoder, self).__init__()

        self.start_id = start_id
        self.end_id = end_id
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding_layer = TransformerEmbedding(d_model, embedding_net)
        self.decoder_layers = nn.ModuleList([
                    MultiAttentiveDecoder(d_model=d_model,
                                        d_enc=d_enc,
                                        d_k=d_k,
                                        d_v=d_v,
                                        d_f=d_f,
                                        n_heads=n_heads,
                                        dropout_prob=dropout_prob) 
                    for _ in range(n_layers)
                ])

        self.out_layer = nn.Sequential(nn.Linear(d_model, d_model//2),
                                        nn.ReLU(),
                                        nn.Linear(d_model//2, d_model//4),
                                        nn.ReLU(),
                                        nn.Linear(d_model//4, self.vocab_size))


    def forward(self, input_batch, encoder_out, context, input_mask, enc_mask):
        assert input_batch.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(input_batch)
        assert encoder_out.dim() == 3, 'Expected tensor with 2 dimensions but got {}'.format(encoder_out)
        assert input_mask.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(input_mask)
        assert enc_mask.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(enc_mask)
        assert input_batch.shape[0] == encoder_out.shape[0], 'Inconsistent batch size'
        assert input_batch.shape[0] == context.shape[0], 'Inconsistent batch size'
        assert input_batch.shape[0] == input_mask.shape[0], 'Inconsistent batch size'
        assert input_batch.shape[0] == enc_mask.shape[0], 'Inconsistent batch size'
        #input mask is the padding mask
        device = input_batch.device
        #self attention mask is a mask resulting from the combination of attention and padding masks
        #the attention mask is an upper triangular mask that avoid each word to attend to the future
        #the padding mask instead is contains 0's for the entries containing padding in the correspondent sequence
        #the resulting matrix avoids each word to attend to future and delete the attention between paddings
        self_attn_mask = torch.tensor((np.triu(np.ones((input_batch.shape[0], input_batch.shape[1], input_batch.shape[1])), k=1) == 0), dtype=torch.long).to(device)
        self_attn_mask &= input_mask[:, :, None]

        #encoder attention mask avoid 2 things:
        # the decoder to attend to encoder padding (to apply row wise)
        # to use the decoder padding as query (to apply column wise)
        #pdb.set_trace()
        enc_attn_mask = torch.zeros((input_mask.shape[0], input_mask.shape[1], enc_mask.shape[1])).to(device)
        enc_attn_mask[:, :] = enc_mask[:, None, :]
        enc_attn_mask.transpose_(1, 2)
        enc_attn_mask[:, :] *= input_mask[:, None, :]
        enc_attn_mask.transpose_(1, 2)

        input_embs = self.embedding_layer(input_batch)
        #pdb.set_trace()
        x = input_embs
        for idx in range(len(self.decoder_layers)):
            x = self.decoder_layers[idx](input_embs=x,  
                                        enc_out=encoder_out,
                                        self_attn_mask=self_attn_mask,
                                        enc_attn_mask=enc_attn_mask)
        pdb.set_trace()


        """
        if self.training:
            for _ in range(gt_sequences.shape[1]):
                #curr_out, (curr_ht, curr_ct) = self.decoder_module(prev_out, (prev_ht, prev_ct))
                #logits = self.out_layer(curr_out)
                #todo save the logits (need to know the sentences lengths)
                #pred_tok = torch.argmax(F.softmax(logits))
                prev_out = gt_sequences['tok_embeddings'][:, 0]
                #prev_ht = curr_ht
                #prev_ct = curr_ct
            #todo end tok at the end
            return None
        else:
            #here beam search
            pass
        """


    def __str__(self):
        return super().__str__()



class TransformerEmbedding(nn.Module):

    def __init__(self, d_model, embedding_net):
        super(TransformerEmbedding, self).__init__()
        assert embedding_net.embedding_dim == d_model, 'Embedding size of {} does not match d_model of {}'.format(embedding_net.embedding_dim, d_model)
        self.embedding_net = embedding_net
        self.d_model = self.embedding_net.embedding_dim
        self.positional_embeddings = self.init_positional(max_seq_len=150,
                                                        emb_dim=self.d_model)


    def forward(self, input_seq):
        assert input_seq.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(input_seq.dim())
        batch_size = input_seq.shape[0]
        input_len = input_seq.shape[1]
        device = input_seq.device
        pos_emb = self.positional_embeddings[:input_len].to(device)
        #enforce the embedding to prevent loose on information by multiplying it with a scalar
        return self.embedding_net(input_seq)*math.sqrt(self.d_model) + pos_emb[None, :, :]


    def init_positional(self, max_seq_len, emb_dim):
        """Create positional embeddings (from "Attention Is All You Need", Vaswani et al. 2017)

        Args:
            n_position (int): number of elements in the sequence
            emb_dim (int): size of embeddings

        Returns:
            toch.FloatTensor: a positional embedding with shape [N_POSITION x EMB_DIM] 
        """
        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (k // 2) / emb_dim) for k in range(emb_dim)]
            if pos != 0 else np.zeros(emb_dim) for pos in range(max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)


    def __str__(self):
        return super().__str__()



class MultiAttentiveDecoder(nn.Module):

    def __init__(self, d_model, d_enc, d_k, d_v, d_f, n_heads, dropout_prob):
        super(MultiAttentiveDecoder, self).__init__()

        #multi head self attention
        #encoder attention
        #fusion layer
        self.multi_head_self = MultiHeadSelfAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, dropout_prob=dropout_prob)
        self.multi_head_enc = MultiHeadEncoderAttention(d_model=d_model, d_enc=d_enc, d_k=d_k, d_v=d_v, n_heads=n_heads, dropout_prob=dropout_prob)
        self.fusion_module = nn.Sequential()

        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.fnn = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_f),
                                nn.ReLU(),
                                nn.Linear(in_features=d_f, out_features=d_model))



    def forward(self, input_embs, enc_out, self_attn_mask, enc_attn_mask):

        self_out = self.multi_head_self(input_embs, self_attn_mask)

        #flow:
        # multi_head_self
        # add, dropout and norm
        # multi_head_enc
        # add, dropout and norm
        # fusion
        # add, dropout and norm
        # fnn
        # add, dropout and norm
        pass


    def __str__(self):
        return super().__str__()



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout_prob):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_heads = n_heads
        self.attn_heads = nn.ModuleList([SelfAttention(d_model, d_k, d_v, dropout_prob) for _ in range(n_heads)])


    def forward(self, input_batch, input_mask):
        outs = torch.stack([attn_head(input_batch, input_mask) for attn_head in self.attn_heads])


    def __str__(self):
        return super().__str__()



class MultiHeadEncoderAttention(nn.Module):
    def __init__(self, d_model, d_enc, d_k, d_v, n_heads, dropout_prob):
        super(MultiHeadEncoderAttention, self).__init__()

        self.n_heads = n_heads
        self.attn_heads = nn.ModuleList([EncoderAttention(d_model, d_enc, d_k, d_v, dropout_prob) for _ in range(n_heads)])


    def __str__(self):
        return super().__str__()



class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout_prob):
        super(SelfAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(d_model, d_k)
        self.K = nn.Linear(d_model, d_k)
        self.V = nn.Linear(d_model, d_v)
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, input_batch, input_mask):

        query = self.Q(input_batch)
        key = self.K(input_batch)
        value = self.V(input_batch)

        attn_logits = torch.matmul(query, torch.transpose(key, -2, -1))/ math.sqrt(self.d_k)
        masked_attn_logits = attn_logits.masked_fill(input_mask==0, -np.inf)
        attn_scores = F.softmax(masked_attn_logits, -1)
        #pad attentions are row filled with 0's. The softmax will then output NaN for these row.
        # the following line just replace NaN values with 0's
        attn_scores[attn_scores != attn_scores] = 0
        attn_scores = self.dropout(attn_scores)
        out = torch.matmul(attn_scores, value)
        return out


    def __str__(self):
        return super().__str__()    



class EncoderAttention(nn.Module):
    def __init__(self, d_model, d_enc, d_k, d_v, dropout_prob):
        super(EncoderAttention, self).__init__()

        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(d_model, d_k)
        self.K = nn.Linear(d_enc, d_k)
        self.V = nn.Linear(d_enc, d_v)
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, decoder_batch, encoder_out, attn_mask):
        
        query = self.Q(decoder_batch)
        key = self.K(encoder_out)
        value = self.V(encoder_out)

        attn_logits = torch.matmul(query, torch.transpose(key, -2, -1))/ math.sqrt(self.d_k)
        masked_attn_logits = attn_logits.masked_fill(attn_mask==0, -np.inf)
        attn_scores = F.softmax(masked_attn_logits, -1)
        #pad attentions are row filled with 0's. The softmax will then output NaN for these row.
        # the following line just replace NaN values with 0's
        attn_scores[attn_scores != attn_scores] = 0
        attn_scores = self.dropout(attn_scores)
        out = torch.matmul(attn_scores, value)
        return out

    def __str__(self):
        return super().__str__()