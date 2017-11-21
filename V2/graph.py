from __future__ import print_function
from matplotlib import cm
from numpy.random import randn

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

def getNumberPOIs(m):
    # extract relevant input/output data pairs
    size=1
    input_pos = []
    maxPOIs=0
    t=0
    for row in m: # number of row in m

        nbrLines=len(row) # Size of the row
        for r in range(nbrLines-size): # In the row between 1 and the size of the row
            inputR=""
            for rr in range(size): #remove the first columns, so the userId
                inputR+=row[rr+r+1]
                if(int(inputR)>int(maxPOIs)):
                    maxPOIs=inputR
                    #print(t)
            input_pos.append(int(inputR))
        t+=1

    print("Number of POIs: "+str(maxPOIs))

    return int(maxPOIs)

def getNumberPOIs2(m):
    # extract relevant input/output data pairs
    input_pos = []
    my_dict = {}
    t=0
    x=0
    for i in m:
        if(t!=0):
            t=True
            for ii in input_pos:
                if(ii==i):
                    t=False
            if(t):
                my_dict[x]=i
                x+=1
                input_pos.append(i)
        t+=1

    maxPOIs= len(input_pos)

    return int(maxPOIs),my_dict

def getKey(m,x):
    for key, value in m.items():
        if(int(value)==int(x)):
            return int(key)
    return 0

def get_data(seperate, timestep):
    inputfile = 'C:\\Users\\Yannick\\Desktop\\Memoire\\Datasets\\mdcdb\\processed_Ids\\user_id_staypoints.csv'
    ifile = open(inputfile, "rU")
    reader = csv.reader(ifile, delimiter=",")

    m, exv = [], []
    rownum = 0
    for row in reader:
        if(rownum==0):
            print("original: "+str(row))
        try:
            newlist = []
            for word in row:
                word = word.split(", ")
                newlist.extend(word)

            m.append(newlist)
        except IndexError:
            exv.append(row)
        rownum += 1

    m = np.array(m)
    print(int(m[0][1]))
    print("-----------------------------")
    print("Time | longitude | latitude")
    print("worked: "+str(m[0]))
    print("-----------------------------")
    print("Good data format: "+str(len(m)))
    print("Bad data format: "+str(len(exv)))
    print("Data number: "+str(len(m)+len(exv)))
    print("Useable: "+str((round(len(m)/(len(m)+len(exv)),2))*100)+" %")
    print("Loose: "+str((round(len(exv)/(len(m)+len(exv)),2))*100)+" %")
    print("Max length: "+str(len(max(m, key=len)))+" by line "+str(max(m, key=len)[0]))
    print("Min length: "+str(len(min(m, key=len)))+" by line "+str(min(m, key=len)[0]))
    print("-----------------------------")

    numberOfPOIs = getNumberPOIs(m)

    inputTest = []
    inputUserTest = []
    for user in m: # 150
        for seq in range(1,len(user)): # 37 to 1670
            inputUserTest.append(int(user[seq]))
    inputUserTest = np.array(inputUserTest)
    inputTest.append(inputUserTest)
    inputTest = np.array(inputTest)
    unique, counts = np.unique(inputTest, return_counts=True)
    print(dict(zip(unique, counts)))
    hist=dict(zip(unique, counts))


    if seperate:
        input = []
        for user in m: # 150
            inputUser = []
            for seq in range(1,len(user)-timestep): # 37 to 1670
                col=[]
                for i in range(numberOfPOIs*timestep):
                    col.append(0)
                for t in range(timestep):
                    n = int(user[seq+t])-1
                    col[n+t*numberOfPOIs]=1
                inputUser.append(col)
            inputUser = np.array(inputUser)
            input.append(inputUser)

        input = np.array(input)
        #print(input)

        output = []
        for user in m: # 150
            outputUser = []
            x=0
            for seq in range(timestep,len(user)): # 37 to 1670
                if(x!=0):
                    col=[]
                    for i in range(numberOfPOIs): # 238
                        col.append(0)
                    n = int(user[seq])-1
                    col[n]=1
                    outputUser.append(col)
                x+=1
            outputUser = np.array(outputUser)
            output.append(outputUser)

        output = np.array(output)
        #print(output)

    else:
        input = []
        inputUser = []
        for user in m: # 150
            for seq in range(1,len(user)-timestep): # 37 to 1670
                col=[]
                for i in range(numberOfPOIs*timestep): # 238
                    col.append(0)
                for t in range(timestep):
                    n = int(user[seq+t])-1
                    col[n+t*numberOfPOIs]=1
                inputUser.append(col)
        inputUser = np.array(inputUser)
        input.append(inputUser)

        input = np.array(input)

        output = []
        outputUser = []
        for user in m: # 150
            x=0
            for seq in range(timestep,len(user)): # 37 to 1670
                if(x!=0):
                    col=[]
                    for i in range(numberOfPOIs): # 238
                        col.append(0)
                    n = int(user[seq])-1
                    col[n]=1
                    outputUser.append(col)
                x+=1
        outputUser = np.array(outputUser)
        output.append(outputUser)

        output = np.array(output)



    return input,output,len(m), numberOfPOIs, hist, m

def heatmap(m, number, size):
    for user in range(0,number):
        # take input
        data = []
        t=m[user]
        numbers=len(t)
        numberOfPOIs, dictPOIs = getNumberPOIs2(t)
        x = 0
        for seq in t: # 37 to 1670
            x+=1
            if(x!=1):
                n = int(getKey(dictPOIs,int(seq)))+1
                data.append(n)
        data = np.array(data)
        data2=np.zeros((size, size))
        for j in range(0, len(data)-1):
            for i in range(1,size+1):
                for n in range(1,size+1):
                    #if(i==n):
                    #   data[i-1][n-1]=-100
                    if((str(data[j])+"|"+str(data[j+1]))==str(i)+"|"+str(n)):
                        data2[i-1][n-1]+=1

        data2=data2/numbers

        # Make plot with vertical (default) colorbar
        fig, ax = plt.subplots()

        data2 = np.array(data2)

        data1 = data2/np.expand_dims(np.sum(data2, axis=1),1)
        plt.imshow(data1, cmap="Spectral")
        plt.colorbar()
        plt.xticks(np.arange(10), np.arange(10)+1)
        plt.yticks(np.arange(10), np.arange(10)+1)
        plt.savefig('C:/Users/Yannick/Desktop/Image1/Graph/Heatmap/user'+str(user)+'size'+str(size)+'Heatmap.png')
        plt.close()
        """max=np.max(data2)
        min=np.min(data2)
        mid=(max-min)/2

        cax = ax.imshow(data2, interpolation='nearest', cmap=cm.coolwarm) #cm.coolwarm or "hot"
        ax.set_title('User: '+str(user)+', Size: '+str(size)+' - Heatmap ')

        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax, ticks=[min, mid, max])
        cbar.ax.set_yticklabels(['< '+str(min), str(mid), '> '+str(max)])  # vertically oriented colorbar
        plt.savefig('C:/Users/Yannick/Desktop/Image1/Graph/Heatmap/user'+str(user)+'size'+str(size)+'Heatmap.png')
        plt.close()"""

def mouvment(m, number):
    for user in range(0,number):
        x=[]
        for i in range(len(m[user])):
            x.append(i)

        fig = plt.figure()
        plt.plot(x, m[user], 'g-')
        plt.axis([0, len(m[user]), 0, 238])
        plt.xlabel('Time')
        plt.ylabel('POIs')
        plt.title('User: '+str(user)+' - Movement between POIs')
        plt.savefig('C:/Users/Yannick/Desktop/Image1/Graph/Movement/user'+str(user)+'Movement.png')
        plt.close(fig)

def allOccurrences(hist, min, max):
    y=[]
    for i in range(min,max):
        y.append(0)
    for key, value in hist.items():
        if(key<=max and key>min):
            y[key-min-1]=value
    x=[]
    for o in range(min,max):
        x.append(o+1)

    y_pos = np.arange(len(x))
    fig = plt.figure()
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.xlabel('POIs')
    plt.ylabel('Occurrences')
    plt.title('POIs and them occurrences')
    plt.savefig('C:/Users/Yannick/Desktop/Image1/Graph/Occurrences/AllOccurrences.png')
    plt.close(fig)

def occurrences(m, c):
    maxs=[]
    mins=[]
    for m in range(1,int(238/c)+2):
        maxs.append(m*c)
        mins.append(m*c-c)
    for n in range(len(maxs)):

        max=maxs[n]
        min=mins[n]
        y=[]
        for i in range(min,max):
            y.append(0)
        for key, value in hist.items():
            if(key<=max and key>min):
                y[key-min-1]=value
        x=[]
        for o in range(min,max):
            x.append(o+1)

        y_pos = np.arange(len(x))
        fig = plt.figure()
        plt.bar(y_pos, y, align='center', alpha=0.5)
        plt.xticks(y_pos, x)
        plt.xlabel('POIs')
        plt.ylabel('Occurrences')
        plt.title('POIs and them occurrences '+str(n))
        plt.savefig('C:/Users/Yannick/Desktop/Image1/Graph/Occurrences/Occurrences'+str(n)+'.png')
        plt.close(fig)

input, output, number, numberOfPOIs, hist, m = get_data(True,1)
"""print("================")
print("Get Data finish")
print("================")
allOccurrences(hist, 0, numberOfPOIs)
print("================")
print("All Occurences finish")
print("================")
occurrences(m, 15)
print("================")
print("Occurences finish")
print("================")
mouvment(m, number)
print("================")
print("Movement finish")
print("================")"""
heatmap(m, number, 10)
print("================")
print("Heatmap finish")
print("================")
