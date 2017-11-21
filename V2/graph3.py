from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv

type=['Linear', 'LSTM', 'NN', 'Logistic']
zone='V3'

for ty in range(1,2):
    plotLossTrain=[]
    plotLossTest=[]
    plotAccuracyTrain=[]
    plotAccuracyTest=[]

    filename1 = 'C:\\Users\\Yannick\\Desktop\\'+zone+'\\'+type[ty]+'PlotLossTrain.npy'
    filename2 = 'C:\\Users\\Yannick\\Desktop\\'+zone+'\\'+type[ty]+'PlotLossTest.npy'
    filename3 = 'C:\\Users\\Yannick\\Desktop\\'+zone+'\\'+type[ty]+'PlotAccuracyTrain.npy'
    filename4 = 'C:\\Users\\Yannick\\Desktop\\'+zone+'\\'+type[ty]+'PlotAccuracyTest.npy'

    plotLossTrain= np.load(filename1)
    plotLossTest= np.load(filename2)
    plotAccuracyTrain=np.load(filename3)
    plotAccuracyTest=np.load(filename4)


    timestep= 5
    # parameters
    learning_rate = 0.0001
    batch_size = 20
    display_step = 1000000
    num_epoch = 1000
    display_epoch = 500
    train_size = 0.7
    num_user = 1 # number of user to procces
    plotAccuracyTestAll=[]
    for j in range(0,timestep):
        print(j)
        # represent the results
        bestAccuracy=[]
        plotEpoch = []
        for i in range(1,num_epoch+1):
            plotEpoch.append(i)

        # for each user wanted we plot a graph for Loss/epoch and Accurac/Epoch
        for i in range(0, num_user):
            print("i: "+str(i))
            print(plotAccuracyTrain.shape)
            print(plotAccuracyTest.shape)
            fig = plt.figure()
            plt.plot(plotEpoch, plotLossTrain[j][i], 'r-', label="Train")
            plt.plot(plotEpoch, plotLossTest[j][i], 'b-', label="Test")
            plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTest[j][i]), np.amin(plotLossTest[j][i])), 'g-', label="Best")
            plt.legend(loc='best')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.title('User: '+str(i)+' Timestep: '+str(j+1)+' - Loss by epoch')
            plt.axis([0, np.amax(plotEpoch), 0, np.amax(plotLossTest[j][i])])
            plt.savefig('C:/Users/Yannick/Desktop/'+zone+'/'+type[ty]+'/user'+str(i)+'Timestep'+str(j+1)+'Loss.png')
            plt.close(fig)

            fig = plt.figure()
            plt.plot(plotEpoch, plotAccuracyTrain[j][i], 'r-', label="Train")
            plt.plot(plotEpoch, plotAccuracyTest[j][i], 'b-', label="Test")
            plt.plot((0, np.amax(plotEpoch)), (np.amax(plotAccuracyTest[j][i]), np.amax(plotAccuracyTest[j][i])), 'g-', label="Best")
            plt.legend(loc='best')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.title('User: '+str(i)+' Timestep: '+str(j+1)+' - Accuracy by epoch')
            plt.axis([0, np.amax(plotEpoch), 0, 1])
            plt.savefig('C:/Users/Yannick/Desktop/'+zone+'/'+type[ty]+'/user'+str(i)+'Timestep'+str(j+1)+'Accuracy.png')
            plt.close(fig)
            bestAccuracy.append(np.amax(plotAccuracyTest[j][i]))
            if(i==num_user-1):
                plotAccuracyTestAll.append(bestAccuracy)

        # only the worst and best accuracy between all users tested
        print("Best: "+str(np.amax(bestAccuracy)))
        print("Worst: "+str(np.amin(bestAccuracy)))

    finalAccuracyByTimestep=[]
    print(plotAccuracyTestAll)
    print(plotAccuracyTestAll[0])

    accuracyTest=[0.0, 0.0, 0.0, 0.0, 0.0]
    accuracyTest=np.array(accuracyTest)
    print(accuracyTest)
    for i in range(0, num_user):
        col=[]
        for j in range(0,timestep):
            col.append(plotAccuracyTestAll[j][i])
            print(accuracyTest[j])
            print(plotAccuracyTestAll[j][i])
            print("---")
            accuracyTest[j]+=plotAccuracyTestAll[j][i]
        finalAccuracyByTimestep.append(col)

        plotTimestep=[]
        for t in range(1,timestep+1):
            plotTimestep.append(t)
        fig = plt.figure()
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(plotTimestep, finalAccuracyByTimestep[i], 'b-', label="Test")
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Timestep')
        plt.title('User: '+str(i)+' - Accuracy by timestep')
        plt.axis([1, timestep, 0, 1])
        plt.savefig('C:/Users/Yannick/Desktop/'+zone+'/'+type[ty]+'/user'+str(i)+'Accuracy2.png')
        plt.close(fig)

    accuracyTest=accuracyTest/num_user
    print(accuracyTest)
    plotTimestep=[]
    for t in range(1,timestep+1):
        plotTimestep.append(t)
    fig = plt.figure()
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(plotTimestep, accuracyTest, 'b-', label="Test")
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Timestep')
    plt.title(' Accuracy average by timestep')
    plt.axis([1, timestep, 0, 1])
    plt.savefig('C:/Users/Yannick/Desktop/'+zone+'/'+type[ty]+'/AccuracyAverage.png')
    plt.close(fig)