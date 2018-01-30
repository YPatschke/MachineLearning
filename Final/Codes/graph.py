from __future__ import print_function
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt

# selection models which we need to plot graph by listing all the their parameters
n_hiddens = 1 # number of hidden layers
n_neurons = 20 # number of neurons per hidden layers
timesteps= 5 # number of timesteps

modelType="LSTM" # you can select 'LSTM', 'Linear', 'Logistic' or 'NN'
path="C:/Users/Yannick/Desktop/finalTest/"

# Parameters
learning_rate = 0.0001
batch_size = 20
num_epoch = 10
num_user = 5 # number of user to procces

# load data
path+=modelType
if modelType == "LSTM" or modelType == "NN":
    path+='/H'+str(n_hiddens)+'N'+str(n_neurons)+'T'+str(timesteps)
else:
    path+='/T'+str(timesteps)
filename1 = path+'/Data/'+modelType+'PlotLossTrain.npy'
filename2 = path+'/Data/'+modelType+'PlotLossTest.npy'
filename3 = path+'/Data/'+modelType+'PlotAccuracyTrain.npy'
filename4 = path+'/Data/'+modelType+'PlotAccuracyTest.npy'
filename5 = path+'/Data/'+modelType+'PlotTrue.npy'
filename6 = path+'/Data/'+modelType+'PlotPrediction.npy'
plotLossTrain= np.load(filename1)
plotLossTest= np.load(filename2)
plotAccuracyTrain=np.load(filename3)
plotAccuracyTest=np.load(filename4)
plotTrue=np.load(filename5)
plotPrediction=np.load(filename6)

# list of the number of epoch to plot
plotEpoch = [i for i in range(1,num_epoch+1)]

# for each user wanted we plot a graph for Loss/epoch and Accurac/Epoch
for user in range(0, num_user):
    fig = plt.figure()
    plt.plot(plotEpoch, plotLossTrain[user], 'r-', label="Train")
    plt.plot(plotEpoch, plotLossTest[user], 'b-', label="Test")
    plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTrain[user]), np.amin(plotLossTrain[user])), 'g-', label="Best")
    plt.legend(loc='best')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('User: '+str(user)+', Timestep: '+str(timesteps)+' - Loss by epoch \n '+' Hidden layer(s): '+str(n_hiddens)+', Neurons:'+str(n_neurons))
    plt.axis([0, np.amax(plotEpoch), 0, np.amax([np.amax(plotLossTrain[user]),np.amax(plotLossTest[user])])])
    plt.savefig(path+'/user'+str(user)+'Timestep'+str(timesteps)+'Hidden'+str(n_hiddens)+'Neurons'+str(n_neurons)+'Loss.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(plotEpoch, plotAccuracyTrain[user], 'r-', label="Train")
    plt.plot(plotEpoch, plotAccuracyTest[user], 'b-', label="Test")
    plt.plot((0, np.amax(plotEpoch)), (np.amax(plotAccuracyTest[user]), np.amax(plotAccuracyTest[user])), 'g-', label="Best")
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('User: '+str(user)+', Timestep: '+str(timesteps)+' - Accuracy by epoch \n '+' Hidden layer(s): '+str(n_hiddens)+', Neurons:'+str(n_neurons))
    plt.axis([0, np.amax(plotEpoch), 0, 1])
    plt.savefig(path+'/user'+str(user)+'Timestep'+str(timesteps)+'Hidden'+str(n_hiddens)+'Neurons'+str(n_neurons)+'Accuracy.png')
    plt.close(fig)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(plotTrue[user], label='True Data')
    plt.plot(plotPrediction[user], label='Prediction')
    plt.legend()
    plt.title('User: '+str(user)+' Timestep: '+str(timesteps)+' - True data vs Prediction (point by point) \n Hidden:'+str(n_hiddens)+' , Neuron:'+str(n_neurons)+' , Learning rate:'+str(learning_rate)+' , Batch size:'+str(batch_size)+' , Epoch:'+str(num_epoch))
    plt.savefig(path+'/Movement/user'+str(user)+'Timestep'+str(timesteps)+'Hidden'+str(n_hiddens)+'Neurons'+str(n_neurons)+'Movement.png')
    plt.close(fig)

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(plotTrue[user], label='True Data')
    plt.plot(plotPrediction[user], label='Prediction')
    plt.legend()
    plt.axis([0, len(plotTrue[user]), 0, 240])
    plt.title('User: '+str(user)+' Timestep: '+str(timesteps)+' - True data vs Prediction (point by point) \n Hidden:'+str(n_hiddens)+' , Neuron:'+str(n_neurons)+' , Learning rate:'+str(learning_rate)+' , Batch size:'+str(batch_size)+' , Epoch:'+str(num_epoch))
    plt.savefig(path+'/Movement2/user'+str(user)+'Timestep'+str(timesteps)+'Hidden'+str(n_hiddens)+'Neurons'+str(n_neurons)+'Movement.png')
    plt.close(fig)