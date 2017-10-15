from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

import loadData as ld
rng = np.random

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
def simple_data_iterator(X, t, batch_size, num_epochs, shuffle):
    # get data sizes
    N = X.shape[0]

    # define ids for shuffle
    ids = np.arange(N)

    for epoch in range(num_epochs):
        if shuffle:
            np.random.shuffle(ids)

        for k in range(int(np.floor(len(ids)/batch_size))):
            i = ids[k*batch_size:(k+1)*batch_size]

            yield X[i], t[i], epoch+1, k+1

# linear model run
def runLinear(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, train_size):

    # split data
    train_input, train_output, test_input, test_output, num_examples = split(input, output, train_size, True)

    # tf Graph Input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # set model weights
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # construct a linear model
    pred = tf.add(tf.multiply(X, W), b)

    # mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_examples)
    # adamOptimzer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:

        # run the initializer
        sess.run(init)

        plotLossTrainUser =[]
        plotLossTestUser=[]
        plotAccuracyTrainUser=[]
        plotAccuracyTestUser=[]

        # fit all training data
        for epoch in range(num_epoch):
            for (x, y) in zip(train_input, train_output):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # display logs per epoch step
            if (epoch % display_epoch == 0 or epoch == 0):
                c = sess.run(cost, feed_dict={X: train_input, Y:train_output})
                print("User: "+str(user)+", Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
                pred_test = sess.run(pred, feed_dict={X:train_input})
                rnd_id = np.random.randint(0, train_input.shape[0])
                print("Pred: "+str(np.argmax(pred_test[rnd_id,:], axis=0)))
                print("Targ: "+str(np.argmax(train_output[rnd_id,:], axis=0)))
                print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_input,Y: test_output}))
                print("Train Accuracy:", sess.run(accuracy, feed_dict={X: train_input,Y: train_output}))
                print("Test Loss= ", sess.run(cost, feed_dict={X: test_input, Y: test_output}))
                print("Train Loss= ", sess.run(cost, feed_dict={X: train_input, Y: train_output}))
                print("----------")

            # stock loss and accuracy at each last step of each epoch
            plotLossTrainUser.append(sess.run(cost, feed_dict={X: train_input, Y: train_output}))
            plotLossTestUser.append(sess.run(cost, feed_dict={X: test_input, Y: test_output}))
            plotAccuracyTrainUser.append(sess.run(accuracy, feed_dict={X: train_input,Y: train_output}))
            plotAccuracyTestUser.append(sess.run(accuracy, feed_dict={X: test_input,Y: test_output}))

        # final accuracy
        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_input, Y: train_output})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    return plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser

# get data
input, output, number, numberOfPOIs = ld.get_data(False)

# variables for stock loss and accuracy at different epoch
plotLossTrain=[]
plotLossTest=[]
plotAccuracyTrain=[]
plotAccuracyTest=[]

# parameters
learning_rate = 0.0001
batch_size = 25
display_step = 1000000
num_epoch = 100
display_epoch = 1
train_size = 0.7
num_user = 1 # number of user to procces

# network parameters
num_input = numberOfPOIs # same vector for all user based and all POIs
num_classes = numberOfPOIs # same vector for all user based and all POIs

# for each user wanted we procced a linear model and stock loss and accuracy
for i in range(0,num_user):
    plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser = runLinear(i, input[i], output[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, train_size)
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
    plt.plot(plotEpoch, plotLossTrain[i], 'r-', label="Train")
    plt.plot(plotEpoch, plotLossTest[i], 'b-', label="Test")
    plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTest[i]), np.amin(plotLossTest[i])), 'g-', label="Best")
    plt.legend(loc='best')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('User: '+str(i)+' - Loss by epoch')
    plt.axis([0, np.amax(plotEpoch), 0, np.amax(plotLossTest[i])])
    plt.savefig('Path/Linear/user'+str(i)+'Loss.png')
    plt.show()

    plt.plot(plotEpoch, plotAccuracyTrain[i], 'r-', label="Train")
    plt.plot(plotEpoch, plotAccuracyTest[i], 'b-', label="Test")
    plt.plot((0, np.amax(plotEpoch)), (np.amax(plotAccuracyTest[i]), np.amax(plotAccuracyTest[i])), 'g-', label="Best")
    plt.legend(loc='best')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('User: '+str(i)+' - Accuracy by epoch')
    plt.axis([0, np.amax(plotEpoch), 0, 1])
    plt.savefig('Path/Linear/user'+str(i)+'Accuracy.png')
    plt.show()
    bestAccuracy.append(np.amax(plotAccuracyTest[i]))

# only the worst and best accuracy between all users tested
print("Best: "+str(np.amax(bestAccuracy)))
print("Worst: "+str(np.amin(bestAccuracy)))
