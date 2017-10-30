from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

import loadData3 as ld
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
def runLinear(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, num_classes, train_size, timestep):
    tf.reset_default_graph()
    # split data
    train_input, train_output, test_input, test_output, num_examples = split(input, output, train_size, False)

    # tf Graph Input
    #X = tf.placeholder("float")
    #Y = tf.placeholder("float")
    X = tf.placeholder(tf.float32, [None, num_input]) # mnist data image of shape 28*28=784
    Y = tf.placeholder(tf.float32, [None, num_classes]) # 0-9 digits recognition => 10 classes

    # Set model weights
    #W = tf.Variable(rng.randn(), name="weight")
    #b = tf.Variable(rng.randn(), name="bias")
    W = tf.Variable(tf.zeros([num_input, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Construct a linear model
    #pred = tf.add(tf.multiply(X, W), b)
    pred = tf.matmul(X, W) + b

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_examples)
    #cost = tf.reduce_mean(tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_examples))

    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
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

        nbr_batch = int(num_examples/batch_size)
        num_steps = nbr_batch # total steps to train

        # get data prepared for each epoch and step
        train_batch_gen = simple_data_iterator(train_input, train_output, batch_size, num_epoch, True)

        plotLossTrainUser =[]
        plotLossTestUser=[]
        plotAccuracyTrainUser=[]
        plotAccuracyTestUser=[]

        # fit all training data
        #for epoch in range(num_epoch):
            #for (x, y) in zip(train_input, train_output):
        for batch_x, batch_y, epoch, step in train_batch_gen:
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            # display logs per epoch step
            if ((epoch % display_epoch == 0 or epoch == 1) and (step % display_step == 0 or step == num_steps)):
                c = sess.run(cost, feed_dict={X: train_input, Y:train_output})
                #print("Timeste: "+str(timestep)+", User: "+str(user)+", Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
                print("Timeste: "+str(timestep)+", User: "+str(user)+", Epoch:", '%04d' % epoch)
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
            if(step==num_steps):
                cost_tr, accurarcy_tr = sess.run([cost, accuracy], feed_dict={X: train_input, Y: train_output})
                cost_te, accurarcy_te = sess.run([cost, accuracy], feed_dict={X: test_input, Y: test_output})
                plotLossTrainUser.append(cost_tr)
                plotLossTestUser.append(cost_te)
                plotAccuracyTrainUser.append(accurarcy_tr)
                plotAccuracyTestUser.append(accurarcy_te)

        # final accuracy
        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_input, Y: train_output})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    return plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser

timestep= 5
plotAccuracyTestAll=[]
for j in range(1,timestep+1):
    # get data
    input, output, number, numberOfPOIs = ld.get_data(True, j)

    # variables for stock loss and accuracy at different epoch
    plotLossTrain=[]
    plotLossTest=[]
    plotAccuracyTrain=[]
    plotAccuracyTest=[]

    # parameters
    learning_rate = 0.0001
    batch_size = 20
    display_step = 1000000
    num_epoch = 1000
    display_epoch = 100
    train_size = 0.7
    num_user = 150 # number of user to procces

    # network parameters
    num_input = numberOfPOIs*j # same vector for all user based and all POIs
    num_classes = numberOfPOIs # same vector for all user based and all POIs

    # for each user wanted we procced a linear model and stock loss and accuracy
    for i in range(0,num_user):
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser = runLinear(i, input[i], output[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, num_classes, train_size, j)
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
        plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTest[i]), np.amin(plotLossTest[i])), 'g-', label="Best")
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('User: '+str(i)+' Timestep: '+str(j)+' - Loss by epoch')
        plt.axis([0, np.amax(plotEpoch), 0, np.amax(plotLossTest[i])])
        plt.savefig('Path/Linear/user'+str(i)+'Timestep'+str(j)+'Loss.png')
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
        plt.savefig('Path/Linear/user'+str(i)+'Timestep'+str(j)+'Accuracy.png')
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
    plt.savefig('Path/Linear/user'+str(i)+'Accuracy2.png')
    plt.close(fig)
