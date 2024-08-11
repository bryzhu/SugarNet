#global constants

#raw data sampling rate (min)
FREQ = 15
#batch size for time series generator
BATCH = 32
#Features for learning
train_feature = ['delta', 'carbs', 'bolus', 'basal', 'bolus_long']
PRETRAIN_EPOCHS = 100
#30
LEARN_EPOCHS = 100
#50
#Run the pipeline how many times
RUNS = 1
#Verbose level of the trainer
VERBOSE = 1

LEARNING_RATE=0.001
MAX_CARBS=200

BOLUS_RATE = 0.014