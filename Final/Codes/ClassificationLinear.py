from __future__ import print_function

import numpy as np
import tensorflow as tf
import split as sp
import data_iterator as di

# Linear model run
def runLinear(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, num_input, num_classes, train_size, timestep, path):
    tf.reset_default_graph()
    # split data
    train_input, train_output, test_input, test_output, num_examples = sp.split(input, output, train_size, False)

    # tf Graph Input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    # Set model weights
    W = tf.Variable(tf.zeros([num_input, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Construct a linear model
    pred = tf.matmul(X, W) + b

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_examples)
    # need to have also a cost for my test set
    num_examples2=num_examples*(1-train_size)/train_size
    cost2 = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_examples2)

    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # define saver opartor
    saver = tf.train.Saver()

    # initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:

        # run the initializer
        sess.run(init)

        # calculate the number of step
        nbr_batch = int(num_examples/batch_size)
        num_steps = nbr_batch # total steps to train

        # get data prepared for each epoch and step for trainning
        train_batch_gen = di.simple_data_iterator3(train_input, train_output, batch_size, num_epoch, True)

        # variable to store my results
        LossTrainUser =[]
        LossTestUser=[]
        AccuracyTrainUser=[]
        AccuracyTestUser=[]
        TrueMovement=[]
        PredictionMovement=[]

        # begin to learn by processing over my training data
        for batch_x, batch_y, epoch, step in train_batch_gen:
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            # display informations sometimes
            if((epoch % display_epoch == 0 or epoch == 1) and (step % display_step == 0 or step == num_steps-1)):
                # calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Timestep: "+str(timestep)+", User: "+str(user)+",Epoch: "+str(epoch)+", Step "+str(step)+", Minibatch Loss= "+str(loss)+", Training Accuracy= "+str(acc))
                cost_train, accurarcy_train = sess.run([cost, accuracy], feed_dict={X: train_input, Y: train_output})
                cost_test, accurarcy_test = sess.run([cost2, accuracy], feed_dict={X: test_input, Y: test_output})
                pred_test_batch = sess.run(pred, feed_dict={X:batch_x})
                pred_test = sess.run(pred, feed_dict={X:test_input})
                print("Predi. on batch: "+str(np.argmax(pred_test_batch, axis=1)))
                print("Target on batch: "+str(np.argmax(batch_y, axis=1)))
                print("Predi. on test data: "+str(np.argmax(pred_test, axis=1)))
                print("Target on test data: "+str(np.argmax(test_output, axis=1)))
                print("Testing Accuracy:"+str(accurarcy_test))
                print("Train Accuracy:"+str(accurarcy_train))
                print("Test Loss= "+str(cost_test))
                print("Train Loss= "+str(cost_train))
                print("----------")

            # at each first step of one epoch: calculate and store losses, accuracy, true movement and predictions movement
            if step == 1:
                cost_train, accurarcy_train = sess.run([cost, accuracy], feed_dict={X: train_input, Y: train_output})
                cost_test, accurarcy_test, pred_test = sess.run([cost2, accuracy, pred], feed_dict={X: test_input, Y: test_output})
                LossTrainUser.append(cost_train)
                LossTestUser.append(cost_test)
                AccuracyTrainUser.append(accurarcy_train)
                AccuracyTestUser.append(accurarcy_test)
                TrueMovement=np.argmax(test_output, axis=1)
                PredictionMovement=np.argmax(pred_test, axis=1)


        saver.save(sess, path+"/Save/"+"U"+str(user)+"T"+str(timestep)+"Model.ckpt")

    print("Optimization Finished!")

    return LossTrainUser, LossTestUser, AccuracyTrainUser, AccuracyTestUser, TrueMovement, PredictionMovement