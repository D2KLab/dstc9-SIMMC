
"""
class SIMMCFashionConfig():
    #? not used
    _FASHION_ACTION_NO = 5
    _FASHION_ATTRS_NO = 33
"""

model_conf = {
    'dropout_prob': .3,
    'hidden_size': 300,
    'freeze_embeddings': False,
    'n_encoders': 1,
    'encoder_heads': 4,
    'n_decoders': 4,
    'decoder_heads': 4
}

special_toks = {
    'pad_token': '[PAD]',
    'start_token': '[START]',
    'end_token': '[END]',
    'unk_token': '[UNK]',
}

train_conf = {
    'seed': 240797,
    'distractors_sampling': -1, #-1 to avoid sampling
    'lr': 1e-3,
    'weight_decay': 0,
    'ckpt_folder': 'mm_response_generation/checkpoints'
}