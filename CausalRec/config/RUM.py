LAMBDA2=0.0001
BATCH_SIZE = 64
MAX_ITER = 1000
EMB_SIZE = 32
ALPHA = 0.2
BETA = 1
LEARNING_RATE = 0.01

TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.2
NEGATIVE_SAMPLING_COUNT = 1

RECOMMEND_NUM = 5
RECOMMEND_NUM_1 = 10

VAL_BATCH_SIZE = 128


ADAM = "ADAM"
SGD = "SGD"
RMSprop = "RMSprop"
OPTIMIZER = RMSprop


# RUMC
GAMMA1 = 7
GAMMA2 = 1
THRESHOLD = 0.3


'''
music instrument
automotive
movie
'''
MUSIC_INSTRUMENT = "music instrument"
AUTOMOTIVE = "automotive"
MOVIE = "movie"
DIGITAL_MUSIC = "digital music"
LUXURY_BEAUTY = "luxury beauty"
OSS = "oss"
MEETUP = "Meetup"
CAMRA2011 = "CAMRa2011"

scenario = LUXURY_BEAUTY

OUTPUT_DIR = 'CausalRec/saved_model/RUM'