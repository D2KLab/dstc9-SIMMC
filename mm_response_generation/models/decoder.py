import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#the value with which to mask the attention
# DO NOT USE '-INF' BECAUSE IT WILL GENERATE NaN AFTER SOFTMAX FOR PADDING ROWS (filled with all 0's)
_MASKING_VALUE=-1e30

class Decoder(nn.Module):

    def __init__(self,
                d_model,
                d_enc,
                d_context,
                d_k,
                d_v,
                d_f,
                n_layers,
                n_heads,
                embedding_net,
                input_vocab_size,
                out_vocab_size,
                dropout_prob):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.out_vocab_size = out_vocab_size
        self.n_layers = n_layers

        self.embedding_layer = TransformerEmbedding(input_vocab_size, d_model)
        #self.embedding_layer = embedding_net
        self.decoder_layers = nn.ModuleList([
                    MultiAttentiveDecoder(d_model=d_model,
                                        d_enc=d_enc,
                                        d_context=d_context,
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
                                        nn.Linear(d_model//4, self.out_vocab_size))


    def forward(self, 
                input_batch, 
                encoder_out, 
                #history_context,
                visual, 
                enc_mask,
                visual_mask,
                input_mask=None):
        device = input_batch.device
        if input_mask is None:
            input_mask = torch.ones(input_batch.shape, dtype=torch.long).to(device)
        assert input_batch.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(input_batch.dim())
        assert encoder_out.dim() == 3, 'Expected tensor with 2 dimensions but got {}'.format(encoder_out.dim())
        assert input_mask.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(input_mask.dim())
        assert enc_mask.dim() == 2, 'Expected tensor with 2 dimensions but got {}'.format(enc_mask.dim())
        assert input_batch.shape[0] == encoder_out.shape[0], 'Inconsistent batch size'
        #assert input_batch.shape[0] == history_context.shape[0], 'Inconsistent batch size'
        #assert input_batch.shape[0] == visual_context.shape[0], 'Inconsistent batch size'
        assert input_batch.shape[0] == input_mask.shape[0], 'Inconsistent batch size'
        assert input_batch.shape[0] == enc_mask.shape[0], 'Inconsistent batch size'
        assert input_batch.shape == input_mask.shape, 'Inconsistent mask size, {} and {}'.format(input_batch.shape, input_mask.shape)
        assert encoder_out.shape[:2] == enc_mask.shape, 'Inconsistent mask size, {} and {}'.format(encoder_out.shape[:2], enc_mask.shape)

        assert input_batch.device == encoder_out.device, 'Different devices'
        #assert input_batch.device == history_context.device, 'Different devices'
        #assert input_batch.device == visual_context.device, 'Different devices'
        assert input_batch.device == input_mask.device, 'Different devices'
        assert input_batch.device == enc_mask.device, 'Different devices'
        #input mask is the padding mask
        #self attention mask is a mask resulting from the combination of attention and padding masks
        #the attention mask is an upper triangular mask that avoid each word to attend to the future
        #the padding mask instead is contains 0's for the entries containing padding in the correspondent sequence
        #the resulting matrix avoids each word to attend to future and delete the attention between paddings
        if input_mask is not None:
            self_attn_mask = torch.tensor((np.triu(np.ones((input_batch.shape[0], input_batch.shape[1], input_batch.shape[1])), k=1) == 0), dtype=torch.long).to(device)
            self_attn_mask &= input_mask[:, :, None]
        else:
            #this is used during inference, where the words are given one by one (future is not present)
            self_attn_mask = torch.ones((input_batch.shape[0], input_batch.shape[1], input_batch.shape[1]))

        #encoder attention mask avoid 2 things:
        # the decoder to attend to encoder padding (to apply row wise)
        # to use the decoder padding as query (to apply column wise)
        enc_attn_mask = torch.zeros((input_mask.shape[0], input_mask.shape[1], enc_mask.shape[1])).to(device)
        enc_attn_mask[:, :] = enc_mask[:, None, :]
        enc_attn_mask = enc_attn_mask.transpose(1, 2)
        enc_attn_mask[:, :] *= input_mask[:, None, :]
        enc_attn_mask = enc_attn_mask.transpose(1, 2)

        visual_attn_mask = torch.zeros((input_mask.shape[0], input_mask.shape[1], visual_mask.shape[1])).to(device)
        visual_attn_mask[:, :] = visual_mask[:, None, :]
        visual_attn_mask = visual_attn_mask.transpose(1, 2)
        visual_attn_mask[:, :] *= input_mask[:, None, :]
        visual_attn_mask = visual_attn_mask.transpose(1, 2)

        x = self.embedding_layer(input_batch)
        """
        x = torch.zeros((input_batch.shape[0], input_batch.shape[1], self.d_model), dtype=torch.float32).to(device)
        #todo x[:, 0] = self.embedding_layer(self.start_id)
        for i in range(input_batch.shape[-1]):
            curr_embs = self.embedding_layer(input=input_batch[:, :i+1],
                                            input_mask=self_attn_mask[:, i, :i+1],
                                            input_token_type=torch.zeros(input_batch[:, :i+1].shape, dtype=torch.long).to(device))
            x[:, i] = curr_embs[:, i]
        """
        #todo from here
        #pdb.set_trace()
        for idx in range(len(self.decoder_layers)):
            #pdb.set_trace()
            x = self.decoder_layers[idx](input_embs=x,  
                                        enc_out=encoder_out,
                                        #history_context=history_context,
                                        visual=visual,
                                        self_attn_mask=self_attn_mask,
                                        enc_attn_mask=enc_attn_mask,
                                        visual_attn_mask=visual_attn_mask)
        #pdb.set_trace()
        vocab_logits = self.out_layer(x)
        return vocab_logits


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

    def __init__(self, vocab_size, d_model, embedding_net=None):
        super(TransformerEmbedding, self).__init__()
        if embedding_net is not None:
            assert embedding_net.embedding_dim == d_model, 'Embedding size of {} does not match d_model of {}'.format(embedding_net.embedding_dim, d_model)
            self.embedding_net = embedding_net
        else:
            self.embedding_net = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.positional_embeddings = self.init_positional(max_seq_len=512,
                                                        emb_dim=d_model)


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

    def __init__(self,
                d_model, 
                d_enc, 
                d_context, 
                d_k, 
                d_v, 
                d_f, 
                n_heads, 
                dropout_prob):
        super(MultiAttentiveDecoder, self).__init__()

        #multi head self attention
        #encoder attention
        #fusion layer
        self.multi_head_self = MultiHeadSelfAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, dropout_prob=dropout_prob)
        self.multi_head_enc = MultiHeadEncoderAttention(d_model=d_model, d_enc=d_enc, d_k=d_k, d_v=d_v, n_heads=n_heads, dropout_prob=dropout_prob)
        self.multi_head_item = MultiHeadEncoderAttention(d_model=d_model, d_enc=d_enc, d_k=d_k, d_v=d_v, n_heads=n_heads, dropout_prob=dropout_prob)
        #self.fusion_module = FusionModule(d_model=d_model, d_context=d_context, dropout_prob=dropout_prob)

        self.layerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_prob)

        self.fnn = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_f),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_prob),
                                nn.Linear(in_features=d_f, out_features=d_model))



    def forward(self, 
                input_embs, 
                enc_out, 
                #history_context, 
                visual,
                visual_attn_mask,
                self_attn_mask, 
                enc_attn_mask):
        #pdb.set_trace()
        self_attn_out = self.multi_head_self(input_embs, self_attn_mask)
        sub_out1 = self.layerNorm(input_embs + self.dropout(self_attn_out))

        enc_attn_out = self.multi_head_enc(sub_out1, enc_out, enc_attn_mask)
        sub_out2 = self.layerNorm(sub_out1 + self.dropout(enc_attn_out))

        item_attn_out = self.multi_head_item(sub_out2, visual, visual_attn_mask)
        sub_out3 = self.layerNorm(sub_out2 + self.dropout(item_attn_out))

        #fusion_out = self.fusion_module(sub_out2, history_context, visual_context)
        #sub_out3 = self.layerNorm(sub_out2 + self.dropout(fusion_out))

        fnn_out = self.fnn(sub_out3) #todo change to subout3
        sub_out4 = self.layerNorm(sub_out3 + self.dropout(fnn_out)) #todo change to subout3

        return sub_out4


    def __str__(self):
        return super().__str__()



class FusionModule(nn.Module):

    def __init__(self, d_model, d_context, dropout_prob):
        super(FusionModule, self).__init__()

        self.d_context = d_context
        self.d_model = d_model
        d_cat = d_model+d_context
        self.h_stream = nn.Sequential(nn.Linear(in_features=d_cat, out_features=d_cat//2),
                                    nn.Linear(in_features=d_cat//2, out_features=d_cat//4),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout_prob),
                                    nn.Linear(in_features=d_cat//4, out_features=d_cat//2),
                                    nn.Linear(in_features=d_cat//2, out_features=d_cat))
        self.v_stream = nn.Sequential(nn.Linear(in_features=d_cat, out_features=d_cat//2),
                                    nn.Linear(in_features=d_cat//2, out_features=d_cat//4),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout_prob),
                                    nn.Linear(in_features=d_cat//4, out_features=d_cat//2),
                                    nn.Linear(in_features=d_cat//2, out_features=d_cat))
        self.fusion_stream = nn.Sequential(nn.Linear(in_features=2*d_cat, out_features=d_cat),
                                            nn.ReLU(),
                                            nn.Dropout(p=dropout_prob),
                                            nn.Linear(in_features=d_cat, out_features=d_model))


    def forward(self, decoder_batch, history_cntx, visual_cntx):
        assert decoder_batch.dim() == 3, 'Expected 3 dimensions, got {}'.format(decoder_batch.dim())
        assert history_cntx.shape[-1] == self.d_context, 'History dimension {} does not match d_context of {}'.format(history_cntx.shape[-1], self.d_context)
        assert history_cntx.shape[-1] == visual_cntx.shape[-1], 'History and visual context sizes do not match'

        h_in = torch.cat((decoder_batch, history_cntx.unsqueeze(1).expand(-1, decoder_batch.shape[1], -1)), dim=-1)
        v_in = torch.cat((decoder_batch, visual_cntx.unsqueeze(1).expand(-1, decoder_batch.shape[1], -1)), dim=-1)
        h_out = self.v_stream(h_in)
        v_out = self.h_stream(v_in)
        fuse_in = torch.cat((h_out, v_out), dim=-1)
        fuse_out = self.fusion_stream(fuse_in)
        return fuse_out



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout_prob):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_heads = n_heads
        self.attn_heads = nn.ModuleList([SelfAttention(d_model, d_k, d_v, dropout_prob) for _ in range(n_heads)])


    def forward(self, input_batch, attn_mask):
        return torch.cat([attn_head(input_batch, attn_mask) for attn_head in self.attn_heads], dim=-1)


    def __str__(self):
        return super().__str__()



class MultiHeadEncoderAttention(nn.Module):
    def __init__(self, d_model, d_enc, d_k, d_v, n_heads, dropout_prob):
        super(MultiHeadEncoderAttention, self).__init__()

        self.n_heads = n_heads
        self.attn_heads = nn.ModuleList([EncoderAttention(d_model, d_enc, d_k, d_v, dropout_prob) for _ in range(n_heads)])


    def forward(self, decoder_batch, encoder_out, attn_mask):
        return torch.cat([attn_head(decoder_batch, encoder_out, attn_mask) for attn_head in self.attn_heads], dim=-1)


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


    def forward(self, input_batch, attn_mask):

        query = self.Q(input_batch)
        key = self.K(input_batch)
        value = self.V(input_batch)

        attn_logits = torch.matmul(query, torch.transpose(key, -2, -1))/ math.sqrt(self.d_k)
        #mask padding and future words with a great negative value.
        # DO NOT USE '-INF' VALUES BECAUSE THEY WILL PRODUCE NaN VALUES AFTER SOFTMAX FOR PADDING WORDS
        masked_attn_logits = attn_logits.masked_fill(attn_mask==0, value=_MASKING_VALUE)
        attn_scores = F.softmax(masked_attn_logits, -1)
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
        #mask padding and future words with a great negative value.
        # DO NOT USE '-INF' VALUES BECAUSE THEY WILL PRODUCE NaN VALUES AFTER SOFTMAX FOR PADDING WORDS
        masked_attn_logits = attn_logits.masked_fill(attn_mask==0, value=_MASKING_VALUE)
        attn_scores = F.softmax(masked_attn_logits, dim=-1)
        attn_scores = self.dropout(attn_scores)
        out = torch.matmul(attn_scores, value)
        return out

    def __str__(self):
        return super().__str__()