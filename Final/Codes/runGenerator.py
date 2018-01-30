import RandomGenerator as rGen
import BestProbabilityGenerator as bGen
import RandomChoiceProbabilityGenerator as cGen
import loadData as ld
import numpy as np

# Parameters
n_hiddens = 1 # number of hidden layers
n_neurons = 20 # number of neurons per hidden layers
timesteps= 5 # number of timestep, so number of previous result took into account
learning_rate = 0.0001 # learning rate (how fast thee model choose to correcte/learn)
batch_size = 20 # number of example we train in same times
display_step = 2000 # if we want to display more often information during training
num_epoch = 10 # number of time that we prcess all the data in trainning
display_epoch = 1000 # if we want to display more often information during training
train_size = 0.7 # part size of data that will be use for training
userModel = 0 # model from a specific user or a global model

model="H"+str(n_hiddens)+"N"+str(n_neurons)+"T"+str(timesteps)
path="C:/Users/Yannick/Desktop/finalTest/LSTM/"+model
stepToGenerate=0 # 0 to generate the same number of step than true data test
users=5
generator_selected = "Choice" # "Random", "Best" or "Choice"

# get data
input, output, number, numberOfPOIs = ld.get_data(True,1)

# Network Parameters
num_input = numberOfPOIs # same vector for all user based and all POIs
num_classes = numberOfPOIs # same vector for all user based and all POIs

trueTrajectories = []
generatedTrajectories = []
for user in range(users):
    if generator_selected == "Random":
        generate_path, true_path = rGen.runRandomGenerator(model, stepToGenerate, user, input[user], output[user], train_size, timesteps, path)
    elif generator_selected == "Best":
        generate_path, true_path = bGen.runBestProbabilityGenerator(model, stepToGenerate, userModel, user, input[user], output[user], n_hiddens, n_neurons, num_input, num_classes, train_size, timesteps, path)
    elif generator_selected == "Choice":
        generate_path, true_path = cGen.runRandomChoiceProbabilityGenerator(model, stepToGenerate, userModel, user, input[user], output[user], n_hiddens, n_neurons, num_input, num_classes, train_size, timesteps, path)
    generatedTrajectories.append(generate_path)
    trueTrajectories.append(true_path)

generatedTrajectories=np.array(generatedTrajectories)
trueTrajectories=np.array(trueTrajectories)
np.save(path+"/"+generator_selected+"GeneratedTrajectories.npy", generatedTrajectories)
np.save(path+"/"+"TrueTrajectories.npy", trueTrajectories)