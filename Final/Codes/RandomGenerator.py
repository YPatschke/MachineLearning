from __future__ import print_function

import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import loadData as ld
import split as sp
import data_iterator as di

# Random Generator
def runRandomGenerator(model, stepToGenerate, user_to_test, input, output, train_size, timestep, path):
    # split data and get all input, output and number of example uses in the train
    train_input, train_output, test_input, test_output, num_examples = sp.split(input, output, train_size, False)

    train_input, train_output = di.simple_data_iterator2(train_input, train_output, timestep)
    test_input, test_output = di.simple_data_iterator2(test_input, test_output, timestep)


    # calculate batch loss and accuracy
    print("Timestep: "+str(timestep)+", User: "+str(user_to_test))

    POIS=[test_output[:timestep]]

    generate_path_random=[]

    if stepToGenerate == 0:
        stepToGenerate = len(test_output[timestep:])
    for p in range(stepToGenerate):
        for t in range(timestep):
            print("Last positions:"+str(np.argmax(POIS[0][t])))
        #Random
        next_random = random.randrange(238)
        print("New position - Random:"+str(next_random))
        generate_path_random.append(next_random)
        print("="*150)

        POIS2=[0 for _ in range(238)]
        POIS2[[next_random][0]]=1
        POIS2=np.array(POIS2)
        POIS= np.append(POIS[0][1:], [POIS2], axis=0)
        POIS=np.array([POIS])

    TrueMovement=np.argmax(test_output, axis=1)

    print("Path generated - Random: "+ str(generate_path_random))

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    plt.plot(generate_path_random, label='Random')
    plt.plot(TrueMovement[timestep:stepToGenerate+timestep], label='Real')
    plt.legend()
    plt.axis([0, len(generate_path_random), 0, 240])
    plt.title("Generated trajectory - Random vs Real \n Movement of user: "+str(user_to_test))
    plt.savefig(path+'/Generate/user'+str(user_to_test)+'Random.png')
    #plt.show(fig)
    plt.close(fig)

    print("Generation Finished!")

    return generate_path_random, TrueMovement[timestep:stepToGenerate+timestep]

