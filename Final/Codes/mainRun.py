import numpy as np

import loadData as ld
import ClassificationLSTM as Clstm
import ClassificationLinear as Clinear
import ClassificationLogistic as Clogistic
import ClassificationNN as Cnn

# order the input and output by users having more transition data, and choose to return only a certain number (num_user)
def orderUsers(input, output, num_user):
    # get only the users with more movement
    inputOrdered = sorted(input, key=lambda x:len(x))
    inputOrdered.reverse()
    inputOrdered=inputOrdered[:num_user]
    outputOrdered = sorted(output, key=lambda x:len(x))
    outputOrdered.reverse()
    outputOrdered=outputOrdered[:num_user]

    return inputOrdered, outputOrdered

# Select the model to use
modelType="LSTM" # you can select 'LSTM', 'Linear', 'Logistic' or 'NN'

# Parameters selection
learning_rate = 0.0001 # learning rate (how fast thee model choose to correcte/learn)
batch_size = 20 # number of example we train in same times
display_step = 2000 # if we want to display more often information during training
num_epoch = 10 # number of time that we prcess all the data in trainning
display_epoch = 500 # if we want to display more often information during training
train_size = 0.7 # part size of data that will be use for training
num_user = 5 # number of user to procces (in our case all user 150 if we want a model per users)
n_hiddens = 1 # number of hidden layers - only useful for NN and LSTM
n_neurons = 20 # number of neurons per hidden layers - only useful for NN and LSTM
timesteps= 5 # number of timestep, so number of previous positions to take into account

# path where store results
path="C:/Users/Yannick/Desktop/finalTest/"+modelType
if modelType == "LSTM" or modelType == "NN":
    path+='/H'+str(n_hiddens)+'N'+str(n_neurons)+'T'+str(timesteps)
else:
    path+='/T'+str(timesteps)

# prepare some list to store results of the different models to execute
plotLossTrain=[]
plotLossTest=[]
plotAccuracyTrain=[]
plotAccuracyTest=[]
plotTrueMovement=[]
plotPredictionMovement=[]

# get and prepare data for LSTM
input, output, number, numberOfPOIs = ld.get_data(True,1)
# get only the users with more movement
inputOrdered, outputOrdered = orderUsers(input, output, num_user)
# Network Parameters
num_input = numberOfPOIs # same vector for all user based and all POIs
num_classes = numberOfPOIs # same vector for all user based and all POIs

# get and prepare data for other model
if modelType != "LSTM":
    # load data for LSTM
    input, output, number, numberOfPOIs = ld.get_data(True,timesteps)
    # get only the users with more movement
    inputOrdered, outputOrdered = orderUsers(input, output, num_user)
    # Network Parameters
    num_input = numberOfPOIs*timesteps # same vector for all user based and all POIs
    num_classes = numberOfPOIs # same vector for all user based and all POIs

# variables for stock loss and accuracy at different epoch
plotLossTrainTimeStep=[]
plotLossTestTimestep=[]
plotAccuracyTrainTimestep=[]
plotAccuracyTestTimestep=[]
plotTrueMovementTimestep=[]
plotPredictionMovementTimestep=[]

# for each user wanted we procced a linear model and stock loss and accuracy
for i in range(0,num_user):
    if modelType == "LSTM":
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser, plotTrueMovementUser, plotPredictionMovementUser = Clstm.runLSTM(i, inputOrdered[i], outputOrdered[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hiddens, n_neurons, num_input, num_classes, train_size, timesteps, path)
    if modelType == "Linear":
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser, plotTrueMovementUser, plotPredictionMovementUser = Clinear.runLinear(i, inputOrdered[i], outputOrdered[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, num_classes, train_size, timesteps, path)
    if modelType == "Logistic":
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser, plotTrueMovementUser, plotPredictionMovementUser = Clogistic.runLogistic(i, inputOrdered[i], outputOrdered[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, num_classes, train_size, timesteps, path)
    if modelType == "NN":
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser, plotTrueMovementUser, plotPredictionMovementUser = Cnn.runNN(i, inputOrdered[i], outputOrdered[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hiddens, n_neurons, num_input, num_classes, train_size, timesteps, path)
    plotLossTrain.append(plotLossTrainUser)
    plotLossTest.append(plotLossTestUser)
    plotAccuracyTrain.append(plotAccuracyTrainUser)
    plotAccuracyTest.append(plotAccuracyTestUser)
    plotTrueMovement.append(plotTrueMovementUser)
    plotPredictionMovement.append(plotPredictionMovementUser)

plotLossTrain=np.array(plotLossTrain)
plotLossTest=np.array(plotLossTest)
plotAccuracyTrain=np.array(plotAccuracyTrain)
plotAccuracyTest=np.array(plotAccuracyTest)
plotTrueMovement=np.array(plotTrueMovement)
plotPredictionMovement=np.array(plotPredictionMovement)

# store the train/test losses, train/test accuracies, true trajectories and predicted trajectories
a = np.array(plotLossTrain)
np.save(path+"/Data/"+modelType+"PlotLossTrain.npy", a)
a = np.array(plotLossTest)
np.save(path+"/Data/"+modelType+"PlotLossTest.npy", a)
a = np.array(plotAccuracyTrain)
np.save(path+"/Data/"+modelType+"PlotAccuracyTrain.npy", a)
a = np.array(plotAccuracyTest)
np.save(path+"/Data/"+modelType+"PlotAccuracyTest.npy", a)
a = np.array(plotTrueMovement)
np.save(path+"/Data/"+modelType+"PlotTrue.npy", a)
a = np.array(plotPredictionMovement)
np.save(path+"/Data/"+modelType+"PlotPrediction.npy", a)