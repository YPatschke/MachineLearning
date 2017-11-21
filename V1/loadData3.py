import numpy as np
import csv


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


def get_data(seperate, timestep):
    inputfile = 'Path\\user_id_staypoints.csv'
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



    return input,output,len(m), numberOfPOIs


"""input,output, number, numberOfPOIs = get_data(False,4)

print(input[0].shape)
#print(input[1].shape)
print(input[0][0])
print(input[0][1])
print(input[0][2])

print(output[0].shape)
#print(output[1].shape)
print(output[0][0])
print(output[0][1])
print(output[0][2])

print(number)"""
