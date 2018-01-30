from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import loadData as ld
import split as sp
import data_iterator as di

# Best Probability Generator
def runBestProbabilityGenerator(model, stepToGenerate, userModel, user_to_test, input, output, n_hiddens, n_neurons, num_input, num_classes, train_size, timestep, path):
    # clears the default graph stack and resets the global default graph
    tf.reset_default_graph()

    # split data and get all input, output and number of example uses in the train
    train_input, train_output, test_input, test_output, num_examples = sp.split(input, output, train_size, False)

    # create tensor for my input and output - [batch-size, timestep, numberOfFeatures] and [batchsize, numberOfClasses]
    X = tf.placeholder(tf.float32, [None, timestep, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    # create multiple LSTMCells of same size (neurons) for each hidden layers that we want
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(n_neurons) for _ in range(n_hiddens)]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, timestep, numberOfClasses]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,inputs=X,dtype=tf.float32)

    weights={}
    for i in range(1, n_hiddens+1):
        if i == 1:
            weights['h1']=tf.Variable(tf.random_normal([num_input, n_neurons]))
        else:
            weights['h'+str(i)]=tf.Variable(tf.random_normal([n_neurons, n_neurons]))
    weights['out']=tf.Variable(tf.random_normal([n_neurons, num_classes]))

    biases={}
    for i in range(1, n_hiddens+1):
        if i == 1:
            biases['b1']=tf.Variable(tf.random_normal([n_neurons]))
        else:
            biases['b'+str(i)]=tf.Variable(tf.random_normal([n_neurons]))
    biases['out']=tf.Variable(tf.random_normal([num_classes]))

    # create logits of our model - so mutliplication of matrice (output from my layers multiply by my weights, then add biases)
    logits = tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']
    # create prediction based on the logit with the softmax function
    prediction = tf.nn.softmax(logits)

    # define saver opartor
    saver = tf.train.Saver()

    #cost_train, accurarcy_train, cost_test, accurarcy_test, TrueMovement, PredictionMovement = 0,0,0,0,0,0
    # start training
    with tf.Session() as sess:
        # initialize variables from file
        #latest_save_file = tf.train.latest_checkpoint("save")
        saver.restore(sess, path +"/Save/U" + str(userModel) + model + "Model.ckpt")

        test_input, test_output = di.simple_data_iterator2(test_input, test_output, timestep)

        # calculate batch loss and accuracy
        print("Hidden: "+str(n_hiddens)+", Neurons: "+str(n_neurons)+", Timestep: "+str(timestep)+", User: "+str(user_to_test))

        POIS=[test_output[:timestep]]

        generate_path_best=[]

        if stepToGenerate == 0:
            stepToGenerate = len(test_output[timestep:])
        for p in range(stepToGenerate):
            for t in range(timestep):
                print("Last positions:"+str(np.argmax(POIS[0][t])))

            #Best
            pred_POIS = sess.run(prediction, feed_dict={X: POIS})
            next_best = int(np.argmax(pred_POIS, axis=1))
            print("New position - Best Probability: "+str(next_best))
            generate_path_best.append(next_best)
            print("="*150)

            POIS2=[0 for _ in range(238)]
            POIS2[[next_best][0]]=1
            POIS2=np.array(POIS2)
            POIS= np.append(POIS[0][1:], [POIS2], axis=0)
            POIS=np.array([POIS])

        TrueMovement=np.argmax(test_output, axis=1)

        print("Path generated - Best: "+ str(generate_path_best))

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.plot(generate_path_best, label='Best Probability')
        plt.plot(TrueMovement[timestep:stepToGenerate+timestep], label='Real')
        plt.legend()
        plt.axis([0, len(generate_path_best), 0, 240])
        plt.title("Generated trajectory - Best Probability vs Real \n Movement of user: "+str(user_to_test))
        plt.savefig(path+'/Generate/user'+str(user_to_test)+'Normal.png')
        #plt.show(fig)
        plt.close(fig)


    print("Generation Finished!")

    return generate_path_best, TrueMovement[timestep:stepToGenerate+timestep]

