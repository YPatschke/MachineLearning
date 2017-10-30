from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

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
def simple_data_iterator(X, t, batch_size, num_epochs, shuffle=True):
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

# create model
def neural_net(x, weights, biases):
    # hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # hidden fully connected layer
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# neural network model run
def runNN(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hidden_1, n_hidden_2, num_input, num_classes, train_size, timestep):
    # split data
    train_input, train_output, test_input, test_output, num_examples = split(input, output, train_size, False)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    # store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        #'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # construct model
    logits = neural_net(X, weights, biases)
    prediction = tf.nn.softmax(logits)

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        nbr_batch = int(num_examples/batch_size)
        num_steps = nbr_batch # Total steps to train

        # get data prepared for each epoch and step
        train_batch_gen = simple_data_iterator(train_input, train_output, batch_size, num_epoch, True)

        plotLossTrainUser =[]
        plotLossTestUser=[]
        plotAccuracyTrainUser=[]
        plotAccuracyTestUser=[]

        # train in batch for a specified number of epochs
        for batch_x, batch_y, epoch, step in train_batch_gen:
            sess.run(train_op, feed_dict={X:batch_x, Y:batch_y})

            if((epoch % display_epoch == 0 or epoch == 1) and (step % display_step == 0 or step == num_steps)):
                # calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Timeste: "+str(timestep)+", User: "+str(user)+", Epoch= "+str(epoch)+", Step= " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                # calculate prediction, loss and accuracy
                pred_test = sess.run(prediction, feed_dict={X:batch_x})
                rnd_id = np.random.randint(0, batch_x.shape[0])
                print("Pred: "+str(np.argmax(pred_test[rnd_id,:], axis=0)))
                print("Targ: "+str(np.argmax(batch_y[rnd_id,:], axis=0)))
                print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_input,Y: test_output}))
                print("Train Accuracy:", sess.run(accuracy, feed_dict={X: train_input,Y: train_output}))
                print("Test Loss= ", sess.run(loss_op, feed_dict={X: test_input, Y: test_output}))
                print("Train Loss= ", sess.run(loss_op, feed_dict={X: train_input, Y: train_output}))
                print("----------")

            if(step==num_steps):
                plotLossTrainUser.append(sess.run(loss_op, feed_dict={X: train_input, Y: train_output}))
                plotLossTestUser.append(sess.run(loss_op, feed_dict={X: test_input, Y: test_output}))
                plotAccuracyTrainUser.append(sess.run(accuracy, feed_dict={X: train_input,Y: train_output}))
                plotAccuracyTestUser.append(sess.run(accuracy, feed_dict={X: test_input,Y: test_output}))

        print("Optimization Finished!")

        # final accuracy
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_input, Y: test_output}))

    return plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser

timestep= 5
plotAccuracyTestAll=[]
for j in range(1,timestep+1):
    # get data
    input, output, number, numberOfPOIs = ld.get_data(True,j)

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
    display_epoch = 500
    train_size = 0.7
    num_user = 150 # number of user to procces

    # Network Parameters
    n_hidden_1 = 20 # 1st layer number of neurons
    n_hidden_2 = 20 # 2nd layer number of neurons
    num_input = numberOfPOIs*j # same vector for all user based and all POIs
    num_classes = numberOfPOIs # same vector for all user based and all POIs

    # for each user wanted we procced a linear model and stock loss and accuracy
    for i in range(0,num_user):
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser = runNN(i, input[i], output[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hidden_1, n_hidden_2, num_input, num_classes, train_size, j)
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
        #plt.plot(plotEpoch, plotLossTest[i], 'b-', label="Test")
        plt.plot((0, np.amax(plotEpoch)), (np.amin(plotLossTrain[i]), np.amin(plotLossTrain[i])), 'g-', label="Best")
        plt.legend(loc='best')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('User: '+str(i)+' Timestep: '+str(j)+' - Loss by epoch')
        plt.axis([0, np.amax(plotEpoch), 0, np.amax(plotLossTrain[i])])
        plt.savefig('Path/NN/user'+str(i)+'Timestep'+str(j)+'Loss.png')
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
        plt.savefig('Path/NN/user'+str(i)+'Timestep'+str(j)+'Accuracy.png')
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
    plt.savefig('Path/NN/user'+str(i)+'Accuracy2.png')
    plt.close(fig)
