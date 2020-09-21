
"""
class SIMMCFashionConfig():
    #? not used
    _FASHION_ACTION_NO = 5
    _FASHION_ATTRS_NO = 33
"""

model_conf = {
    'dropout_prob': 0,
    'freeze_bert': True,
    'n_decoders': 1,
    'decoder_heads': 4 #todo must be a perfect divisor of 768
}

special_toks = {
    'pad_token': '[PAD]',
    'start_token': '[CLS]',
    'end_token': '[SEP]',
    'unk_token': '[UNK]',
}

train_conf = {
    'seed': 240797,
    'distractors_sampling': -1, #-1 to avoid sampling
    'lr': 1e-3,
    'weight_decay': 0,
    'ckpt_folder': 'mm_response_generation/checkpoints'
}