from __future__ import print_function
from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt



type='LSTM'
folder='Image1'

plotLossTrain=[]
plotLossTest=[]
plotAccuracyTrain=[]
plotAccuracyTest=[]
plotTrue=[]
plotPrediction=[]

filename1 = 'C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Data/'+type+'PlotLossTrain.npy'
filename2 = 'C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Data/'+type+'PlotLossTest.npy'
filename3 = 'C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Data/'+type+'PlotAccuracyTrain.npy'
filename4 = 'C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Data/'+type+'PlotAccuracyTest.npy'
filename5 = 'C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Data/'+type+'PlotTrue.npy'
filename6 = 'C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Data/'+type+'PlotPrediction.npy'

plotLossTrain= np.load(filename1)
plotLossTest= np.load(filename2)
plotAccuracyTrain=np.load(filename3)
plotAccuracyTest=np.load(filename4)
plotTrue=np.load(filename5)
plotPrediction=np.load(filename6)

n_hiddens = [1,5]
n_neurons = [1,10]
timesteps= [1,2]

# Parameters
learning_rate = 0.01
batch_size = 20
display_step = 20000
num_epoch = 1000
display_epoch = 500
train_size = 0.7
num_user = 150 # number of user to procces

print("="*100)
print(plotLossTrain.shape)
print(plotLossTest.shape)
print(plotAccuracyTrain.shape)
print(plotAccuracyTest.shape)
print(plotTrue.shape)
print(plotPrediction.shape)

# represent the results
bestAccuracy=[]
lastAccuracy=[]

plotEpoch = []
for i in range(1,num_epoch+1):
    plotEpoch.append(i)

bestAccuracy=[]
lastAccuracy=[]
for hidden in range(0,len(n_hiddens)):
    bestAccuracyHidden=[]
    lastAccuracyHidden=[]
    for neuron in range(0,len(n_neurons)):
        bestAccuracyNeuron=[]
        lastAccuracyNeuron=[]
        for timestep in range(0,len(timesteps)):
            bestAccuracyTimestep=[]
            lastAccuracyTimestep=[]
            # for each user wanted we plot a graph for Loss/epoch and Accurac/Epoch
            for user in range(0, num_user):
                fig = plt.figure()
                plt.plot(plotEpoch, plotLossTrain[hidden][neuron][timestep][user], 'r-', label="Train")
                plt.plot(plotEpoch, plotLossTest[hidden][neuron][timestep][user], 'b-', label="Test")
                plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTrain[hidden][neuron][timestep][user]), np.amin(plotLossTrain[hidden][neuron][timestep][user])), 'g-', label="Best")
                plt.legend(loc='best')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.title('User: '+str(user)+', Timestep: '+str(timesteps[timestep])+' - Loss by epoch \n '+' Hidden layer(s): '+str(n_hiddens[hidden])+', Neurons:'+str(n_neurons[neuron]))
                plt.axis([0, np.amax(plotEpoch), 0, np.amax([np.amax(plotLossTrain[hidden][neuron][timestep][user]),np.amax(plotLossTest[hidden][neuron][timestep][user])])])
                plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/user'+str(user)+'Timestep'+str(timesteps[timestep])+'Hidden'+str(n_hiddens[hidden])+'Neurons'+str(n_neurons[neuron])+'Loss.png')
                plt.close(fig)

                fig = plt.figure()
                plt.plot(plotEpoch, plotAccuracyTrain[hidden][neuron][timestep][user], 'r-', label="Train")
                plt.plot(plotEpoch, plotAccuracyTest[hidden][neuron][timestep][user], 'b-', label="Test")
                plt.plot((0, np.amax(plotEpoch)), (np.amax(plotAccuracyTest[hidden][neuron][timestep][user]), np.amax(plotAccuracyTest[hidden][neuron][timestep][user])), 'g-', label="Best")
                plt.legend(loc='best')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.title('User: '+str(user)+', Timestep: '+str(timesteps[timestep])+' - Accuracy by epoch \n '+' Hidden layer(s): '+str(n_hiddens[hidden])+', Neurons:'+str(n_neurons[neuron]))
                plt.axis([0, np.amax(plotEpoch), 0, 1])
                plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/user'+str(user)+'Timestep'+str(timesteps[timestep])+'Hidden'+str(n_hiddens[hidden])+'Neurons'+str(n_neurons[neuron])+'Accuracy.png')
                plt.close(fig)

                bestAccuracyTimestep.append(np.amax(plotAccuracyTest[hidden][neuron][timestep][user]))
                lastAccuracyTimestep.append(plotAccuracyTest[hidden][neuron][timestep][user][num_epoch-1])

            bestAccuracyNeuron.append(bestAccuracyTimestep)
            lastAccuracyNeuron.append(lastAccuracyTimestep)

        bestAccuracyHidden.append(bestAccuracyNeuron)
        lastAccuracyHidden.append(lastAccuracyNeuron)

    bestAccuracy.append(bestAccuracyHidden)
    lastAccuracy.append(lastAccuracyHidden)

x=[i for i in range(len(timesteps)+1)]
for timestep in range(0,len(timesteps)):
    for neuron in range(0,len(n_neurons)):
        for hidden in range(0,len(n_hiddens)):
            for user in range(0, num_user):
                fig = plt.figure(facecolor='white')
                ax = fig.add_subplot(111)
                ax.plot(plotTrue[hidden][neuron][timestep][user], label='True Data')
                plt.plot(plotPrediction[hidden][neuron][timestep][user], label='Prediction')
                plt.legend()
                plt.title('User: '+str(user)+' Timestep: '+str(timesteps[timestep])+' - True data vs Prediction (point by point) \n Hidden:'+str(n_hiddens[hidden])+' , Neuron:'+str(n_neurons[neuron])+' , Learning rate:'+str(learning_rate)+' , Batch size:'+str(batch_size)+' , Epoch:'+str(num_epoch))
                plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Movement/user'+str(user)+'Timestep'+str(timesteps[timestep])+'Hidden'+str(n_hiddens[hidden])+'Neurons'+str(n_neurons[neuron])+'Movement.png')
                plt.close(fig)

x=[i for i in range(len(timesteps)+1)]
for timestep in range(0,len(timesteps)):
    for neuron in range(0,len(n_neurons)):
        for hidden in range(0,len(n_hiddens)):
            for user in range(0, num_user):
                fig = plt.figure(facecolor='white')
                ax = fig.add_subplot(111)
                ax.plot(plotTrue[hidden][neuron][timestep][user], label='True Data')
                plt.plot(plotPrediction[hidden][neuron][timestep][user], label='Prediction')
                plt.legend()
                plt.axis([0, len(plotTrue[hidden][neuron][timestep][user]), 0, 240])
                plt.title('User: '+str(user)+' Timestep: '+str(timesteps[timestep])+' - True data vs Prediction (point by point) \n Hidden:'+str(n_hiddens[hidden])+' , Neuron:'+str(n_neurons[neuron])+' , Learning rate:'+str(learning_rate)+' , Batch size:'+str(batch_size)+' , Epoch:'+str(num_epoch))
                plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Movement2/user'+str(user)+'Timestep'+str(timesteps[timestep])+'Hidden'+str(n_hiddens[hidden])+'Neurons'+str(n_neurons[neuron])+'Movement.png')
                plt.close(fig)


accuracyBest=[]
accuracyBest=[[[sum(bestAccuracy[hidden][neuron][timestep])/num_user for timestep in range(0,len(timesteps))] for neuron in range(0,len(n_neurons))] for hidden in range(0,len(n_hiddens))]
accuracyBest=np.array(accuracyBest)
print(accuracyBest.shape)

accuracyLast=[]
accuracyLast=[[[sum(lastAccuracy[hidden][neuron][timestep])/num_user for timestep in range(0,len(timesteps))] for neuron in range(0,len(n_neurons))] for hidden in range(0,len(n_hiddens))]
accuracyLast=np.array(accuracyLast)
print(accuracyLast.shape)

plotHidden=[]
for t in range(len(n_hiddens)):
    plotHidden.append(t)

plotNeuron=[]
for t in range(len(n_neurons)):
    plotNeuron.append(t)

plotTimestep=[]
for t in range(len(timesteps)):
    plotTimestep.append(t)

x=[i for i in range(len(n_hiddens))]
for neuron in range(0,len(n_neurons)):
    for timestep in range(0,len(timesteps)):
        fig = plt.figure()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(plotHidden, accuracyBest[:, neuron, timestep], 'b-', label="Best")
        plt.plot(plotHidden, accuracyLast[:, neuron, timestep], 'g-', label="Last")
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Hidden layer')
        plt.title('Accuracy average by hidden layer \n Neuron:'+str(n_neurons[neuron])+' , Timestep:'+str(timesteps[timestep]))
        plt.axis([0, len(n_hiddens)-1, 0, 1])
        plt.xticks(x, n_hiddens)
        #plt.yticks([0,1,2], [0,0.5,1.0])
        plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Hidden/AccuracyAverageHiddenNeuron'+str(n_neurons[neuron])+'Timestep'+str(timesteps[timestep])+'.png')
        plt.close(fig)

x=[i for i in range(len(n_neurons))]
for hidden in range(0,len(n_hiddens)):
    for timestep in range(0,len(timesteps)):
        fig = plt.figure()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(plotNeuron, accuracyBest[hidden, :, timestep], 'b-', label="Best")
        plt.plot(plotNeuron, accuracyLast[hidden, :, timestep], 'g-', label="Last")
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Neuron')
        plt.title('Accuracy average by neuron \n Hidden:'+str(n_hiddens[hidden])+' , Timestep:'+str(timesteps[timestep]))
        plt.axis([0, len(n_neurons)-1, 0, 1])
        plt.xticks(x, n_neurons)
        #plt.yticks([0,1,2], [0,0.5,1.0])
        plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Neuron/AccuracyAverageNeuronHidden'+str(n_hiddens[hidden])+'Timestep'+str(timesteps[timestep])+'.png')
        plt.close(fig)

x=[i for i in range(len(timesteps))]
for hidden in range(0,len(n_hiddens)):
    for neuron in range(0,len(n_neurons)):
        fig = plt.figure()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(plotTimestep, accuracyBest[hidden, neuron, :], 'b-', label="Best")
        plt.plot(plotTimestep, accuracyLast[hidden, neuron, :], 'g-', label="Last")
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Timestep')
        plt.title('Accuracy average by timestep \n Hidden:'+str(n_hiddens[hidden])+' , Neuron:'+str(n_neurons[neuron]))
        plt.axis([0, len(timesteps)-1, 0, 1])
        plt.xticks(x, timesteps)
        #plt.yticks([0,1,2], [0,0.5,1.0])
        plt.savefig('C:/Users/Yannick/Desktop/'+folder+'/'+type+'/Timestep/AccuracyAverageTimestepHidden'+str(n_hiddens[hidden])+'Neuron'+str(n_neurons[neuron])+'.png')
        plt.close(fig)