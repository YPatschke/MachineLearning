from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt

import loadData3 as ld

# separate data between train and test - Need input, output, train size (percentage) and if we want to shuffle
def split(X, t, train_size, shuffle):
    # get he number of examples use for train
    num_examples = round(len(X)*train_size)
    # get data sizes
    N = X.shape[0]

    # define ids for shuffle
    ids = np.arange(N)

    # shuffle if asked
    if shuffle:
        np.random.shuffle(ids)

    # ids for train and test after shuffle it
    train_ids = ids[:num_examples]
    test_ids = ids[num_examples:]

    # train input and output
    train_input =X[train_ids]
    train_output =t[train_ids]
    # test input and test ouput
    test_input =X[test_ids]
    test_output =t[test_ids]

    return train_input, train_output, test_input, test_output, num_examples

# split data by batch for each epoch and shuffle if asked
def simple_data_iterator2(X, t, num_timesteps):

    def stack_timesteps_block(X, t, num_timesteps, stride=1):
        X_stacked = np.stack([X[k:k+num_timesteps,:] for k in range(0, X.shape[0]-num_timesteps+1, stride)], axis=0)
        t_stacked = t[num_timesteps-1::stride]
        return X_stacked, t_stacked

    X_stacked, t_stacked = stack_timesteps_block(X, t, num_timesteps)

    return X_stacked, t_stacked

# split data by batch for each epoch and shuffle if asked
def simple_data_iterator(X, t, batch_size, num_epochs, num_timesteps, shuffle=True):

    def stack_timesteps_block(X, t, num_timesteps, stride=1):
        X_stacked = np.stack([X[k:k+num_timesteps,:] for k in range(0, X.shape[0]-num_timesteps+1, stride)], axis=0)
        t_stacked = t[num_timesteps-1::stride]
        return X_stacked, t_stacked

    X_stacked, t_stacked = stack_timesteps_block(X, t, num_timesteps)

    # randomize training inputs
    ids = np.arange(X_stacked.shape[0])
    if shuffle:
        np.random.shuffle(ids)

    # yield data in batches
    for epoch in range(num_epochs):
        for step in range(int(np.floor(len(ids)/batch_size))):
            ids_batch = ids[step*batch_size:(step+1)*batch_size]

            yield X_stacked[ids_batch], t_stacked[ids_batch], epoch, step

def RNN(x, weights, biases, timestep, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timestep, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# LSTM model run
def runLSTM(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hidden_1, n_hidden_2, num_input, num_classes, train_size, timestep):
    tf.reset_default_graph()
    # split data
    train_input, train_output, test_input, test_output, num_examples = split(input, output, train_size, False)

    # tf Graph input
    X = tf.placeholder("float", [None, timestep, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Construct a linear model
    logits = RNN(X, weights, biases, timestep, n_hidden_1)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:

        # run the initializer
        sess.run(init)

        nbr_batch = int(num_examples/batch_size)
        num_steps = nbr_batch # total steps to train

        # get data prepared for each epoch and step
        train_batch_gen = simple_data_iterator(train_input, train_output, batch_size, num_epoch, timestep, True)

        train_input, train_output = simple_data_iterator2(train_input, train_output, timestep)
        test_input, test_output = simple_data_iterator2(test_input, test_output, timestep)

        plotLossTrainUser =[]
        plotLossTestUser=[]
        plotAccuracyTrainUser=[]
        plotAccuracyTestUser=[]

        # fit all training data
        #for epoch in range(num_epoch):
        #for (x, y) in zip(train_input, train_output):
        for batch_x, batch_y, epoch, step in train_batch_gen:
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step == 0:
                #Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Timestep: "+str(timestep)+", User: "+str(user)+",Epoch: "+str(epoch)+", Step "+str(step)+", Minibatch Loss= "+str(loss)+", Training Accuracy= "+str(acc))
                cost_tr, accurarcy_tr = sess.run([loss_op, accuracy], feed_dict={X: train_input, Y: train_output})
                cost_te, accurarcy_te = sess.run([loss_op, accuracy], feed_dict={X: test_input, Y: test_output})
                pred_test = sess.run(prediction, feed_dict={X:batch_x})
                pred_test2 = sess.run(prediction, feed_dict={X:test_input})
                rnd_id = np.random.randint(0, batch_x.shape[0])
                print("Pred: "+str(np.argmax(pred_test[rnd_id,:], axis=0)))
                print("Targ: "+str(np.argmax(batch_y[rnd_id,:], axis=0)))
                print("Pred2: "+str(np.argmax(pred_test2, axis=1)))
                print("Targ2: "+str(np.argmax(test_output, axis=1)))
                print("Testing Accuracy:"+str(accurarcy_te))
                print("Train Accuracy:"+str(accurarcy_tr))
                print("Test Loss= "+str(cost_te))
                print("Train Loss= "+str(cost_tr))
                print("----------")
                plotLossTrainUser.append(cost_tr)
                plotLossTestUser.append(cost_te)
                plotAccuracyTrainUser.append(accurarcy_tr)
                plotAccuracyTestUser.append(accurarcy_te)

    print("Optimization Finished!")

    return plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser


timestep= 5

plotAccuracyTestAll=[]
for j in range(1,timestep+1):
    # get data
    input, output, number, numberOfPOIs = ld.get_data(True,1)

    # variables for stock loss and accuracy at different epoch
    plotLossTrain=[]
    plotLossTest=[]
    plotAccuracyTrain=[]
    plotAccuracyTest=[]

    # Parameters
    learning_rate = 0.0001
    batch_size = 20
    display_step = 20000
    num_epoch = 1000
    display_epoch = 10
    train_size = 0.7
    num_user = 150 # number of user to procces

    # Network Parameters
    n_hidden_1 = 20 # 1st layer number of neurons
    n_hidden_2 = 20 # 2nd layer number of neurons
    num_input = numberOfPOIs # same vector for all user based and all POIs
    num_classes = numberOfPOIs # same vector for all user based and all POIs

    # for each user wanted we procced a linear model and stock loss and accuracy
    for i in range(0, num_user):
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser = runLSTM(i, input[i], output[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hidden_1, n_hidden_2, num_input, num_classes, train_size, j)
        plotLossTrain.append(plotLossTrainUser)
        plotLossTest.append(plotLossTestUser)
        plotAccuracyTrain.append(plotAccuracyTrainUser)
        plotAccuracyTest.append(plotAccuracyTestUser)

    # represent the results
    bestAccuracy=[]
    plotEpoch = []
    for i in range(1,num_epoch+1):
        plotEpoch.append(i)
    # for each user wanted we plot a graph for Loss/epoch and Accurac/Epoch
    for i in range(0, num_user):
        fig = plt.figure()
        plt.plot(plotEpoch, plotLossTrain[i], 'r-', label="Train")
        plt.plot(plotEpoch, plotLossTest[i], 'b-', label="Test")
        plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTrain[i]), np.amin(plotLossTrain[i])), 'g-', label="Best")
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('User: '+str(i)+' Timestep: '+str(j)+' - Loss by epoch')
        plt.axis([0, np.amax(plotEpoch), 0, np.amax(plotLossTrain[i])])
        plt.savefig('C:/Users/Yannick/Desktop/Image1/LSTM/user'+str(i)+'Timestep'+str(j)+'Loss.png')
        plt.close(fig)

        fig = plt.figure()
        plt.plot(plotEpoch, plotAccuracyTrain[i], 'r-', label="Train")
        plt.plot(plotEpoch, plotAccuracyTest[i], 'b-', label="Test")
        plt.plot((0, np.amax(plotEpoch)), (np.amax(plotAccuracyTest[i]), np.amax(plotAccuracyTest[i])), 'g-', label="Best")
        plt.legend(loc='best')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('User: '+str(i)+' Timestep: '+str(j)+' - Accuracy by epoch')
        plt.axis([0, np.amax(plotEpoch), 0, 1])
        plt.savefig('C:/Users/Yannick/Desktop/Image1/LSTM/user'+str(i)+'Timestep'+str(j)+'Accuracy.png')
        plt.close(fig)
        bestAccuracy.append(np.amax(plotAccuracyTest[i]))
        if(i==num_user-1):
            plotAccuracyTestAll.append(bestAccuracy)

    # only the worst and best accuracy between all users tested
    print("Best: "+str(np.amax(bestAccuracy)))
    print("Worst: "+str(np.amin(bestAccuracy)))

finalAccuracyByTimestep=[]
print(plotAccuracyTestAll)
print(plotAccuracyTestAll[0])
for i in range(0, num_user):
    col=[]
    for j in range(0,timestep):
        col.append(plotAccuracyTestAll[j][i])
    finalAccuracyByTimestep.append(col)

    plotTimestep=[]
    for t in range(1,timestep+1):
        plotTimestep.append(t)
    fig = plt.figure()
    plt.plot(plotTimestep, finalAccuracyByTimestep[i], 'b-', label="Test")
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Timestep')
    plt.title('User: '+str(i)+' - Accuracy by timestep')
    plt.axis([1, timestep, 0, 1])
    plt.savefig('C:/Users/Yannick/Desktop/Image1/LSTM/user'+str(i)+'Accuracy2.png')
    plt.close(fig)