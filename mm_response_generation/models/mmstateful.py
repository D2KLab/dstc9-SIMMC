import math
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .embednets import ItemEmbeddingNetwork, WordEmbeddingNetwork
from .decoder import Decoder
from .old_encoder import SingleEncoder

_MAX_INFER_LEN = 100

class MMStatefulLSTM(nn.Module):

    def __init__(self,
                word_embeddings_path,
                word2id,
                pad_token,
                start_token,
                end_token,
                unk_token,
                seed,
                dropout_prob,
                hidden_size,
                n_encoders,
                encoder_heads,
                n_decoders,
                decoder_heads,
                freeze_embeddings,
                beam_size=None,
                retrieval_eval=False,
                mode='train',
                device='cpu'):

        torch.manual_seed(seed)
        super(MMStatefulLSTM, self).__init__()

        self.mode = mode
        self.beam_size = beam_size
        self.retrieval_eval = retrieval_eval
        self.start_id = word2id[start_token]
        self.end_id = word2id[end_token]
        self.pad_id = word2id[pad_token]
        
        #self.item_embeddings_layer = ItemEmbeddingNetwork(item_embeddings_path)
        self.word_embeddings_layer = WordEmbeddingNetwork(word_embeddings_path=word_embeddings_path, 
                                                        word2id=word2id, 
                                                        pad_token=pad_token, 
                                                        unk_token=unk_token,
                                                        freeze=freeze_embeddings)
        self.emb_dim = self.word_embeddings_layer.embedding_size
        self.encoder = SingleEncoder(input_size=self.emb_dim,
                                    hidden_size=hidden_size,
                                    dropout_prob=dropout_prob,
                                    encoder_heads=encoder_heads,
                                    embedding_net=self.word_embeddings_layer)

        #for h heads: d_k == d_v == emb_dim/h
        self.decoder = Decoder(d_model=self.emb_dim,
                                d_enc=2*hidden_size,
                                d_context=hidden_size,
                                d_k=self.emb_dim//decoder_heads,
                                d_v=self.emb_dim//decoder_heads,
                                d_f=self.emb_dim//2,
                                n_layers=n_decoders,
                                n_heads=decoder_heads,
                                embedding_net=self.word_embeddings_layer,
                                vocab_size=len(word2id),
                                dropout_prob=dropout_prob,
                                start_id=self.start_id,
                                end_id=self.end_id)


    def forward(self, utterances, utterances_mask, history, actions, attributes, focus_items, 
                candidates_pool, pools_padding_mask, seq_lengths=None):
        """The visual context is a list of visual contexts (a batch). Each visual context is, in turn, a list
            of items. Each item is a list of (key, values) pairs, where key is a tensor containing the word ids
            for the field name and values is a list of values where each value is a tensor of word ids.

        type(visual_context[<sample_in_batch>][<item_n>][<field_n>]) -> tuple(tensor(key), [tensor(value)])

        Args:
            utterances ([type]): [description]
            history ([type]): [description]
            visual_context ([type]): [description]
            seq_lengths ([type], optional): [description]. Defaults to None.
            device (str, optional): [description]. Defaults to 'cpu'.

        Returns:
            [type]: [description]
        """
        #check batch size consistency (especially when using different gpus) and move list tensors to correct gpu
        if self.mode == 'inference':
            assert utterances.shape[0] == 1, 'Only unitary batches allowed during inference'
            assert self.beam_size is not None, 'Beam size need to be defined during inference'
        assert utterances.shape[0] == utterances_mask.shape[0], 'Inconstistent batch size'
        assert utterances.shape[0] == len(history), 'Inconsistent batch size'
        assert utterances.shape[0] == len(actions), 'Inconsistent batch size'
        assert utterances.shape[0] == len(attributes), 'Inconsistent batch size'
        assert utterances.shape[0] == len(focus_items), 'Inconsistent batch size'
        assert utterances.shape[0] == candidates_pool.shape[0], 'Inconsistent batch size'
        assert utterances.shape[0] == pools_padding_mask.shape[0], 'Inconsistent batch size'
        assert utterances.shape == utterances_mask.shape, 'Inconsistent mask size'
        assert candidates_pool.shape == pools_padding_mask.shape, 'Inconsistent mask size'
        curr_device = utterances.device
        for idx, _ in enumerate(history):
            if len(history[idx]):
                history[idx] = history[idx].to(curr_device)
        for idx, _ in enumerate(focus_items):
            focus_items[idx][0] = focus_items[idx][0].to(curr_device)
            focus_items[idx][1] = focus_items[idx][1].to(curr_device)

        u_t_all, v_t_tilde, h_t_tilde = self.encoder(utterances=utterances,
                                                    history=history,
                                                    focus_items=focus_items,
                                                    seq_lengths=seq_lengths)
        #decoding phase
        if self.mode == 'train':
            vocab_logits = self.decoder(input_batch=candidates_pool,
                                        encoder_out=u_t_all,
                                        history_context=h_t_tilde,
                                        visual_context=v_t_tilde,
                                        input_mask=pools_padding_mask,
                                        enc_mask=utterances_mask)
            return vocab_logits
        else:
            #at inference time (NOT EVAL)
            #pdb.set_trace()
            self.never_ending = 0
            dec_args = {'encoder_out': u_t_all, 'history_context': h_t_tilde, 'visual_context': v_t_tilde, 'enc_mask': utterances_mask}
            best_dict = self.beam_search(curr_seq=[self.start_id],
                                        curr_score=0,
                                        dec_args=dec_args,
                                        best_dict={'seq': [], 'score': -float('inf')},
                                        device=curr_device)
            #print('Never-ending generated sequences: {}'.format(self.never_ending))
            #pdb.set_trace()
            infer_res = (best_dict['seq'], best_dict['score'])
            if self.retrieval_eval:
                #eval on retrieval task 
                #build a fake batch by extenpanding the tensors
                vocab_logits = [
                                    self.decoder(input_batch=pool,
                                                encoder_out=u_t_all.expand(pool.shape[0], -1, -1),
                                                history_context=h_t_tilde.expand(pool.shape[0], -1),
                                                visual_context=v_t_tilde.expand(pool.shape[0], -1),
                                                input_mask=pool_mask,
                                                enc_mask=utterances_mask.expand(pool.shape[0], -1))
                                    for pool, pool_mask in zip(candidates_pool, pools_padding_mask)
                                ]
                #candidates_scores shape: Bx100
                candidates_scores = self.compute_candidates_scores(candidates_pool, vocab_logits)
                infer_res += (candidates_scores,)
            return infer_res

    
    def beam_search(self, curr_seq, curr_score, dec_args, best_dict, device):
        #pdb.set_trace()
        if curr_seq[-1] == self.end_id:
            assert len(curr_seq)-1 != 0, 'Division by 0 for generated sequence {}'.format(curr_seq)
            #discard the start_id only. The probability of END token has to be taken into account instead.
            norm_score = curr_score/(len(curr_seq)-1)
            if norm_score > best_dict['score']:
                best_dict['score'], best_dict['seq'] = curr_score, curr_seq[1:].copy() #delete the start_token
            return best_dict
        elif len(curr_seq) > _MAX_INFER_LEN:
            #print('Generated sequence {} beyond the maximum length of {}'.format(curr_seq, _MAX_INFER_LEN))
            self.never_ending += 1
            return best_dict
        vocab_logits = self.decoder(input_batch=torch.tensor(curr_seq).unsqueeze(0).to(device), **dec_args).squeeze(0)
        #take only the prediction for the last word
        vocab_logits = vocab_logits[-1]
        beam_ids = torch.argsort(vocab_logits, descending=True, dim=-1)[:self.beam_size].tolist()
        lprobs = F.log_softmax(vocab_logits, dim=-1)
        for curr_id in beam_ids:
            curr_lprob = lprobs[curr_id].item()
            best_dict = self.beam_search(curr_seq=curr_seq+[curr_id], 
                                        curr_score=curr_score+curr_lprob, 
                                        dec_args=dec_args, 
                                        best_dict=best_dict, 
                                        device=device)
        return best_dict


    def compute_candidates_scores(self, candidates_pools, vocab_logits):
        """The score of each candidate is the sum of the log-likelihood of each word, normalized by its length.
        The score will be a negative value, longer sequences will be penalized without the normalization by length.
        """
        scores = torch.zeros(candidates_pools.shape[:2])
        for batch_idx, (pool_ids, pool_logits) in enumerate(zip(candidates_pools[:, :, 1:], vocab_logits)):
            pool_lprobs = F.log_softmax(pool_logits, dim=-1)
            for sentence_idx, (candidate_ids, candidate_lprobs) in enumerate(zip(pool_ids, pool_lprobs)):
                curr_lprob = []
                for candidate_word, words_probs in zip(candidate_ids, candidate_lprobs):
                    #until padding
                    if candidate_word.item() == self.pad_id:
                        break
                    curr_lprob.append(words_probs[candidate_word.item()].item())
                scores[batch_idx, sentence_idx] = sum(curr_lprob)/len(curr_lprob)
        return scores


    def collate_fn(self, batch):
        """
        This method prepares the batch for the LSTM: padding + preparation for pack_padded_sequence

        Args:
            batch (tuple): tuple of element returned by the Dataset.__getitem__()

        Returns:
            dial_ids (list): list of dialogue ids
            turns (list): list of dialogue turn numbers
            seq_tensor (torch.LongTensor): tensor with BxMAX_SEQ_LEN containing padded sequences of user transcript sorted by descending effective lengths
            seq_lenghts: tensor with shape B containing the effective length of the correspondant transcript sequence
            actions (torch.Longtensor): tensor with B shape containing target actions
            attributes (torch.Longtensor): tensor with Bx33 shape containing attributes one-hot vectors, one for each sample.
        """
        dial_ids = [item[0] for item in batch]
        turns = [item[1] for item in batch]
        transcripts = [torch.tensor(item[2]) for item in batch]
        history = [item[3] for item in batch]
        actions = [item[4] for item in batch]
        attributes = [item[5] for item in batch]
        focus_items = [item[6] for item in batch]
        responses_pool = [item[7] for item in batch]

        assert len(transcripts) == len(dial_ids), 'Batch sizes do not match'
        assert len(transcripts) == len(turns), 'Batch sizes do not match'
        assert len(transcripts) == len(history), 'Batch sizes do not match'
        assert len(transcripts) == len(actions), 'Batch sizes do not match'
        assert len(transcripts) == len(attributes), 'Batch sizes do not match'
        assert len(transcripts) == len(focus_items), 'Batch sizes do not match'
        assert len(transcripts) == len(responses_pool), 'Batch sizes do not match'

        # reorder the sequences from the longest one to the shortest one.
        # keep the correspondance with the target
        transcripts_lengths = torch.tensor(list(map(len, transcripts)), dtype=torch.long)
        transcripts_tensor = torch.zeros((len(transcripts), transcripts_lengths.max()), dtype=torch.long)
        transcripts_padding_mask = torch.zeros((len(transcripts), transcripts_lengths.max()), dtype=torch.long)
        for idx, (seq, seqlen) in enumerate(zip(transcripts, transcripts_lengths)):
            transcripts_tensor[idx, :seqlen] = seq.clone().detach()
            transcripts_padding_mask[idx, :seqlen] = 1

        #pad the history
        padded_history = []
        for history_sample in history:
            if not len(history_sample):
                padded_history.append(history_sample)
                continue
            history_lens = torch.tensor(list(map(len, history_sample)), dtype=torch.long)
            history_tensor = torch.zeros((len(history_sample), history_lens.max()), dtype=torch.long)
            for idx, (seq, seqlen) in enumerate(zip(history_sample, history_lens)):
                history_tensor[idx, :seqlen] = seq.clone().detach()
            padded_history.append(history_tensor)

        #pad the response candidates
        # if training take only the true response (the first one)
        if self.training or not self.retrieval_eval:
            responses_pool = [pool_sample[0] for pool_sample in responses_pool]
            batch_lens = torch.tensor(list(map(len, responses_pool)), dtype=torch.long)
            pools_tensor = torch.zeros((len(responses_pool), batch_lens.max()+2), dtype=torch.long)
            pools_padding_mask = torch.zeros((len(responses_pool), batch_lens.max()+2), dtype=torch.long)
            pools_tensor[:, 0] = self.start_id
            for batch_idx, (seq, seqlen) in enumerate(zip(responses_pool, batch_lens)):
                pools_tensor[batch_idx, 1:seqlen+1] = seq.clone().detach()
                pools_tensor[batch_idx, seqlen+1] = self.end_id
                pools_padding_mask[batch_idx, :seqlen+2] = 1
        else:
            batch_lens = torch.tensor([list(map(len, pool_sample)) for pool_sample in responses_pool], dtype=torch.long)
            pools_tensor = torch.zeros((len(responses_pool), len(responses_pool[0]), batch_lens.max()+2), dtype=torch.long)
            pools_padding_mask = torch.zeros((len(responses_pool), len(responses_pool[0]), batch_lens.max()+2), dtype=torch.long)
            pools_tensor[:, :, 0] = self.start_id
            for batch_idx, (pool_lens, pool_sample) in enumerate(zip(batch_lens, responses_pool)):       
                for pool_idx, (seq, seqlen) in enumerate(zip(pool_sample, pool_lens)):
                    pools_tensor[batch_idx, pool_idx, 1:seqlen+1] = seq.clone().detach()
                    pools_tensor[batch_idx, pool_idx, seqlen+1] = self.end_id
                    pools_padding_mask[batch_idx, pool_idx, :seqlen+2] = 1

        #pad focus items
        padded_focus = []
        keys = [[datum[0] for datum in item] for item in focus_items]
        vals = [[datum[1] for datum in item] for item in focus_items]
        batch_klens = [list(map(len, key)) for key in keys]
        batch_vlens = [list(map(len, val)) for val in vals]
        klens_max, vlens_max = 0, 0
        for klens, vlens in zip(batch_klens, batch_vlens):
            curr_kmax = max(klens)
            curr_vmax = max(vlens)
            klens_max = curr_kmax if curr_kmax > klens_max else klens_max
            vlens_max = curr_vmax if curr_vmax > vlens_max else vlens_max

        for batch_idx, item in enumerate(focus_items):
            curr_keys = torch.zeros((len(item), klens_max), dtype=torch.long)
            curr_vals = torch.zeros((len(item), vlens_max), dtype=torch.long)
            for item_idx, (k, v) in enumerate(item):
                curr_keys[item_idx, :batch_klens[batch_idx][item_idx]] = k.clone().detach()
                curr_vals[item_idx, :batch_vlens[batch_idx][item_idx]] = v.clone().detach()
            padded_focus.append([curr_keys, curr_vals])

        """
        # sort instances by sequence length in descending order and order targets to keep the correspondance
        transcripts_lengths, perm_idx = transcripts_lengths.sort(0, descending=True)
        transcripts_tensor = transcripts_tensor[perm_idx]
        transcripts_padding_mask = transcripts_padding_mask[perm_idx]
        pools_tensor = pools_tensor[perm_idx]
        pools_padding_mask = pools_padding_mask[perm_idx]
        sorted_dial_ids = []
        sorted_dial_turns = []
        sorted_dial_history = []
        sorted_actions = []
        sorted_attributes = []
        sorted_focus_items = []
        for idx in perm_idx:
            sorted_dial_ids.append(dial_ids[idx])
            sorted_dial_turns.append(turns[idx])
            sorted_dial_history.append(padded_history[idx])
            sorted_actions.append(actions[idx])
            sorted_attributes.append(attributes[idx])
            sorted_focus_items.append(padded_focus[idx])
        """

        batch_dict = {}
        batch_dict['utterances'] = transcripts_tensor
        batch_dict['utterances_mask'] = transcripts_padding_mask
        batch_dict['history'] = padded_history
        batch_dict['actions'] = actions
        batch_dict['attributes'] = attributes
        batch_dict['focus_items'] = padded_focus
        #batch_dict['seq_lengths'] = transcripts_lengths

        return dial_ids, turns, batch_dict, pools_tensor, pools_padding_mask


    def __str__(self):
        return super().__str__()







