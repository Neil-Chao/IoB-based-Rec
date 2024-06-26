LAMBDA2=0.001
BATCH_SIZE = 64
MAX_ITER = 400
EMB_SIZE = 32
LEARNING_RATE = 0.005
BERT_MAX_LEN = 6


TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2
NEGATIVE_SAMPLING_COUNT = 1

RECOMMEND_NUM = 5
RECOMMEND_NUM_1 = 30

VAL_BATCH_SIZE = 1024


ADAM = "ADAM"
SGD = "SGD"
OPTIMIZER = ADAM


'''
music instrument
automotive
movie
'''
MUSIC_INSTRUMENT = "music instrument"
DIGITAL_MUSIC = "digital music"
LUXURY_BEAUTY = "luxury beauty"

scenario = LUXURY_BEAUTY

OUTPUT_DIR = 'CausalRec/saved_model/BERT4Rec'