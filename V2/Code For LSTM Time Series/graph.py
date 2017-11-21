from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

filename1 = 'C:\\Users\\Yannick\\Desktop\\V2\\dataTrue.npy'
filename2 = 'C:\\Users\\Yannick\\Desktop\\V2\\dataPredicted.npy'
filename3 = 'C:\\Users\\Yannick\\Desktop\\V2\\dataPredicted2.npy'

dataTrue= np.load(filename1)
dataPredicted= np.load(filename2)
dataPredicted2=np.load(filename3)


seq_len = 1
num_users=150
seqPred=10
for j in range(0, seq_len):
    for user in range(0, num_users):
        print("timestep: "+str(j+5)+", User: "+str(user))
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(dataTrue[j][user], label='True Data')
        plt.plot(dataPredicted2[j][user], label='Prediction')
        plt.legend()
        plt.title('User: '+str(user)+' Timestep: '+str(j+5)+' - True data vs Prediction (point by point)')
        plt.savefig('C:/Users/Yannick/Desktop/V2/LSTM2/user'+str(user)+'Timestep'+str(j+5)+'Graph.png')
        plt.close(fig)

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(dataTrue[j][user], label='True Data')
        #Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(dataPredicted[j][user]):
            padding = [None for p in range(i * seqPred)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        #plt.show()
        plt.title('User: '+str(user)+' Timestep: '+str(j+5)+' - True data vs Prediction (next seq)')
        plt.savefig('C:/Users/Yannick/Desktop/V2/LSTM2/user'+str(user)+'Timestep'+str(j+5)+'GraphMultiple.png')
        plt.close(fig)