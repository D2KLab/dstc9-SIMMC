import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SingleEncoder(nn.Module):

    def __init__(self,
                input_size,
                hidden_size,
                dropout_prob,
                encoder_heads,
                embedding_net):
        super(SingleEncoder, self).__init__()
        self.word_embeddings_layer = embedding_net
        self.sentence_encoder = SentenceEncoder(emb_dim=input_size, 
                                        hidden_size=hidden_size, 
                                        bidirectional=True,
                                        dropout_prob=dropout_prob)

        self.memory_net = MemoryNet(in_features=hidden_size, 
                                    memory_hidden_size=hidden_size, 
                                    dropout_prob=dropout_prob)

        #for h heads: d_k == d_v == emb_dim/h
        self.triton_attention = Triton(in_features=hidden_size,
                                        d_k=hidden_size//encoder_heads,
                                        d_v=hidden_size//encoder_heads,
                                        d_f=hidden_size//2,
                                        n_heads=encoder_heads,
                                        n_layers=1,
                                        dropout_prob=dropout_prob)

        self.layerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, utterances, history, focus_items, seq_lengths=None):
        # todo use attributes to enforce the user utterance?
        # u_t shape [BATCH_SIZE x 2MEMORY_HIDDEN_SIZE], u_t_all shape [BATCH_SIZE x MAX_SEQ_LEN x 2MEMORY_HIDDEN_SIZE]
        embedded_utt_tensor = self.word_embeddings_layer(utterances)
        u_t, u_t_all = self.sentence_encoder(embedded_utt_tensor, seq_lengths)

        #h_t shape h_t[<sample_in_batch>].shape = [N_TURNSx300] or [] if no history
        embedded_hist_tensor = []
        for history_sample in history:
            if not len(history_sample):
                embedded_hist_tensor.append([])
            else:
                embedded_hist_tensor.append(self.word_embeddings_layer(history_sample))
        h_t = []
        for h_embedding in embedded_hist_tensor:
            if not len(h_embedding):
                h_t.append([])
            else:
                h_t.append(self.sentence_encoder(h_embedding)[0])

        #embedded_vis_tensor shape v_t[<sample_in_batch>][0/1].shape = [N_FIELDSx300]
        #v_t shape [<sample_in_batch>]x300
        #v_t_tilde contains contextual embedding for each item in the batch. Shape [Bx300]
        v_t = self.encode_v_context(focus_items)
        v_t_tilde = torch.stack([self.triton_attention(utt, v[0], v[1]) for utt, v in zip(u_t, v_t)])

        # compute history context
        h_t_tilde = []
        for idx in range(u_t.shape[0]):
            if not len(h_t[idx]):
                h_t_tilde.append(u_t[idx])
            else:
                h_t_tilde.append(self.memory_net(u_t[idx], h_t[idx]))
        h_t_tilde = torch.stack(h_t_tilde)

        return u_t_all, v_t_tilde, h_t_tilde


    def encode_v_context(self, focus_images):
        v_batch = []
        for keys, values in focus_images:
            k_ht, _ = self.sentence_encoder(self.word_embeddings_layer(keys))
            v_ht, _ = self.sentence_encoder(self.word_embeddings_layer(values))
            v_batch.append([k_ht, v_ht])
        return v_batch


    def __str__(self):
        return super().__str__()



class SentenceEncoder(nn.Module):

    def __init__(self, emb_dim, hidden_size, dropout_prob, bidirectional=False):
        super(SentenceEncoder, self).__init__()

        self.encoder = nn.LSTM(emb_dim,
                            hidden_size, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        in_features = 2*hidden_size if bidirectional else hidden_size
        self.mlp = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layerNorm = nn.LayerNorm(hidden_size)


    def forward(self, seq, seq_lengths=None):

        if seq_lengths is not None:
            # pack padded sequence
            input_seq = pack_padded_sequence(seq, seq_lengths, batch_first=True) #seq_lengths.cpu().numpy()
        else:
            input_seq = seq
        #to call every forward if DataParallel is used. Otherwise only once inside __init__()
        self.encoder.flatten_parameters()
        sentences_outs, (h_t, c_t) = self.encoder(input_seq)
        #concat right and left hidden states of the last layer
        bidirectional_h_t = torch.cat((h_t[-2], h_t[-1]), dim=-1)
        if seq_lengths is not None:
            # unpack padded sequence
            sentences_outs, input_sizes = pad_packed_sequence(sentences_outs, batch_first=True)
        mlp_out = self.mlp(bidirectional_h_t)
        out = self.layerNorm(self.dropout(mlp_out))

        return out, sentences_outs



class MemoryNet(nn.Module):

    def __init__(self, in_features, memory_hidden_size, dropout_prob):
        super(MemoryNet, self).__init__()

        self.memory_hidden_size = memory_hidden_size
        self.query_encoder = nn.Linear(in_features=in_features, out_features=memory_hidden_size)
        self.memory_encoder = nn.Linear(in_features=in_features, out_features=memory_hidden_size)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layerNorm = nn.LayerNorm(memory_hidden_size)


    def forward(self, input, context, device='cpu'):
        query = self.query_encoder(input)
        memory = self.memory_encoder(context) + self.get_positional_embeddings(context.shape[0], self.memory_hidden_size).to(context.device)

        attn_logits = torch.matmul(query, torch.transpose(memory, 0, 1))/math.sqrt(self.memory_hidden_size)
        attn_scores = F.softmax(attn_logits, -1)
        weighted_memory = torch.sum(attn_scores[:, None] * memory, dim=0)

        out = self.layerNorm(query + self.dropout(weighted_memory))
        return out


    def get_positional_embeddings(self, n_position, emb_dim):
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
            if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
        return torch.from_numpy(position_enc).type(torch.FloatTensor)


# Triton, trident? Not self attention! Triplet as input q, k, v belonging to different conceptual sets
class Triton(nn.Module):
    
    def __init__(self, in_features, d_k, d_v, d_f, n_heads, n_layers, dropout_prob):
        super(Triton, self).__init__()

        assert n_layers >= 1, 'Not acceptable number of layers: {}'.format(n_layers)
        self.n_layers = n_layers
        #self.encoders = nn.ModuleList([TritonEncoder(emb_dim, d_k, d_v, n_heads) for _ in range(n_layers)])
        #encoders = [TritonEncoder(emb_dim, d_k, d_v, d_f, n_heads) for _ in range(n_layers)]
        #self.encoders = nn.Sequential(*encoders)
        #todo change to allow multiple layers. Problem: sequential take only 1 input, so pack inputs to a tuple.
        self.encoders = TritonEncoder(in_features, d_k, d_v, d_f, n_heads, dropout_prob)


    def forward(self, ut, kt, vt):
        out = self.encoders(ut, kt, vt)
        return out


    def __str__(self):
        return super().__str__()



class TritonEncoder(nn.Module):


    def __init__(self, in_features, d_k, d_v, d_f, n_heads, dropout_prob):
        super(TritonEncoder, self).__init__()

        self.multihead_attn = TritonMultiHeadCrossAttention(in_features, d_k, d_v, n_heads)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.layerNorm = nn.LayerNorm(in_features)
        self.fnn = nn.Sequential(nn.Linear(in_features=in_features, out_features=d_f),
                                nn.ReLU(),
                                nn.Linear(in_features=d_f, out_features=in_features))


    def forward(self, u_t, k_t, v_t):
        multihead_out = self.multihead_attn(u_t, k_t, v_t)
        # residual connection is performed after the dropout and before normalization in (Vaswani et al.)
        norm_out = self.layerNorm(self.dropout(multihead_out))
        enc_out = self.fnn(norm_out)
        out = self.layerNorm(self.dropout(multihead_out))
        return out
  
    
    def __str__(self):
        return super().__str__()



class TritonMultiHeadCrossAttention(nn.Module):
    
    def __init__(self, in_features, d_k, d_v, n_heads):
        super(TritonMultiHeadCrossAttention, self).__init__()

        self.n_heads = n_heads
        self.attn_heads = nn.ModuleList([TritonCrossAttentionHead(in_features, d_k, d_v) for _ in range(n_heads)])


    def forward(self, u_t, k_t, v_t):
        attn_outs = []
        heads_out = torch.cat([attn_head(u_t, k_t, v_t) for attn_head in self.attn_heads])
        return heads_out


    def __str__(self):
        return super().__str__()



class TritonCrossAttentionHead(nn.Module):

    def __init__(self, in_features, d_k, d_v):
        super(TritonCrossAttentionHead, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.Q = nn.Linear(in_features, d_k)
        self.K = nn.Linear(in_features, d_k)
        self.V = nn.Linear(in_features, d_v)


    def forward(self, u_t, k_t, v_t):
        query = self.Q(u_t)
        key = self.K(k_t)
        value = self.V(v_t)

        attn_logits = torch.matmul(query, torch.transpose(key, 0, 1))/ math.sqrt(self.d_k)
        attn_scores = F.softmax(attn_logits, -1)
        out = torch.sum(attn_scores[:, None] * value, dim=0)
        return out



    def __str__(self):
        return super().__str__()
