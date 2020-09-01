

"""
class SIMMCFashionConfig():
    #? not used
    _FASHION_ACTION_NO = 5
    _FASHION_ATTRS_NO = 33
"""

class TrainConfig():

    _SEED = 240797
    _DISTRACTORS_SAMPLING = -1 #put to -1 to avoid sampling
    _LEARNING_RATE = 1e-3
    _WEIGHT_DECAY = 1e-1
    _PAD_TOKEN = '[PAD]'
    _UNK_TOKEN = '[UNK]'
    _CHECKPOINT_FOLDER = 'mm_response_generation/checkpoints'