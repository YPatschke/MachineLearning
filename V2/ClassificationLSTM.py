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
def simple_data_iterator(X, t, num_epochs, batch_size, num_timesteps, shuffle=True):
    def stack_timesteps_block(X, t, num_timesteps, stride=1):
        X_stacked = np.stack([X[k:k+num_timesteps,:] for k in range(0, X.shape[0]-num_timesteps+1, stride)], axis=0)
        t_stacked = t[num_timesteps-1::stride]
        return X_stacked, t_stacked

    X_stacked, t_stacked = stack_timesteps_block(X, t, num_timesteps)

    # randomize training inputs
    ids = np.arange(X_stacked.shape[0])
    if shuffle:
        np.random.shuffle(ids)

    # print(ids.shape)
    # print(X_stacked.shape)
    # print(t_stacked.shape)

    # yield data in batches
    for epoch in range(num_epochs):
        for step in range(int(np.floor(len(ids)/batch_size))):
            ids_batch = ids[step*batch_size:(step+1)*batch_size]

            yield X_stacked[ids_batch], t_stacked[ids_batch], epoch, step

class LSTM_Model:
    def __init__(self, num_hidden, num_classes, num_timesteps):
        # save model parameters
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps

        # define lstm cell
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)
        self.state = None

        # define variables
        self.W = tf.Variable(tf.zeros([num_hidden, num_classes], dtype=tf.float32))
        self.b = tf.Variable(tf.zeros([num_classes], dtype=tf.float32))

        # debug variables
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def evaluate(self, x, batch_size, reset=False):
        # determine batch size
        #batch_size = int(x.shape[0])

        # create list of inputs based on the number of timesteps
        x = tf.unstack(x, self.num_timesteps, 1)

        # define zero state to reset memory
        if reset:
            self.state = self.lstm_cell.zero_state(batch_size, tf.float32)

        # compute outputs and states
        output, self.state = tf.contrib.rnn.static_rnn(
            self.lstm_cell, x,
            initial_state=self.state,
            dtype=tf.float32)

        # linear activation
        return tf.matmul(output[-1], self.W) + self.b

    def train(self, x, t, learning_rate, batch_size):
        # evaluate prediction based on input
        p = self.evaluate(x, batch_size, reset=True)

        # define loss
        loss = tf.reduce_sum(tf.square(t - p))

        # define training operator
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

        return self.global_step, train_opt, loss

def compute_score(p, t):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(p - t), axis=1)))

# linear model run
def runLSTM(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hidden_1, num_input, num_classes, train_size, timestep):
    tf.reset_default_graph()
    # split data
    train_input, train_output, test_input, test_output, num_examples = split(input, output, train_size, False)
    print(train_input.shape)
    print(test_input.shape)
    # define inputs
    X = tf.placeholder(tf.float32, [None, timestep, num_input])
    t = tf.placeholder(tf.float32, [None, num_classes])

    # define model
    lstm = LSTM_Model(n_hidden_1, num_classes, timestep)

    # define output
    prediction = lstm.evaluate(X, batch_size, reset=True)

    # define training operations
    global_step, train_opt, loss = lstm.train(X, t, learning_rate, batch_size)

    # define score operations
    score = compute_score(prediction, t)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # launch session
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        if restore_model:
            # initialize variables from file
            latest_save_file = tf.train.latest_checkpoint("save")
            #saver.restore(sess, latest_save_file)
        else:
            sess.run(init_op)  # initialize operatores

        nbr_batch = int(num_examples/batch_size)
        num_steps = nbr_batch # Total steps to train

        # load data
        X_data, t_data, number, numberOfPOIs = ld.get_data(True,j)

        # setup test data generator
        test_batch_gen = simple_data_iterator(train_input, train_output, num_epoch, batch_size, timestep, True)
        train_batch_gen = simple_data_iterator(test_input , test_output, num_epoch, batch_size, timestep, True)

        plotLossTrainUser =[]
        plotLossTestUser=[]
        plotAccuracyTrainUser=[]
        plotAccuracyTestUser=[]

        # train in batch for a specified number of epochs
        for X_batch, t_batch, epoch, step in train_batch_gen:
            _, loss_val, train_score, prediction_val = sess.run(
                [train_opt, loss, score, prediction],
                feed_dict={X:X_batch, t:t_batch})

            if step == 0:
                # print debug log
                print("="*50)
                print(str(user)+">> Epoch: %d, step: %d" % (epoch, step))
                print("> Training:")
                print("loss: %.3f" % loss_val)
                print("mean error distance: %.3f" % train_score)
                print("learning_rate: %.5f" % learning_rate)
                print("> Testing:")
                X_test, t_test, _, _ = next(test_batch_gen)
                test_score, pred_test = sess.run([score, prediction], feed_dict={X:X_test, t:t_test})
                rnd_id = np.random.randint(0, t_test.shape[0])
                print("mean error distance: %.3f" % test_score)
                print("prediction (indice of the POI): "+str(np.argmax(pred_test[rnd_id,:], axis=0)))
                print("target (indice of the POI): "+str(np.argmax(t_test[rnd_id,:], axis=0)))
                """pred_test2 = sess.run(prediction2, feed_dict={X:test_input})
                print("Pred2: "+str(np.argmax(pred_test2, axis=1)))
                print("Targ2: "+str(np.argmax(test_output, axis=1)))"""
                #print("prediction: 0:%.3f, 1:%.3f, 2:%.3f, 3:%.3f, 4:%.3f, 5:%.3f, 6:%.3f, 7:%.3f, 8:%.3f, 9:%.3f, 10:%.3f, 11:%.3f, 12:%.3f, 13:%.3f, 14:%.3f, 15:%.3f, 16:%.3f, 17:%.3f, 18:%.3f, 19:%.3f, 20:%.3f, 21:%.3f, 22:%.3f, 23:%.3f, 24:%.3f, 25:%.3f, 26:%.3f, 27:%.3f ,28:%.3f, 29:%.3f" % tuple(pred_test[rnd_id,:]))
                #print("target:     0:%.3f, 1:%.3f, 2:%.3f, 3:%.3f, 4:%.3f, 5:%.3f, 6:%.3f, 7:%.3f, 8:%.3f, 9:%.3f, 10:%.3f, 11:%.3f, 12:%.3f, 13:%.3f, 14:%.3f, 15:%.3f, 16:%.3f, 17:%.3f, 18:%.3f, 19:%.3f, 20:%.3f, 21:%.3f, 22:%.3f, 23:%.3f, 24:%.3f, 25:%.3f, 26:%.3f, 27:%.3f ,28:%.3f, 29:%.3f" % tuple(t_test[rnd_id,:]))
                plotLossTrainUser.append(sess.run(score, feed_dict={X:X_batch, t:t_batch}))
                plotLossTestUser.append(sess.run(score, feed_dict={X:X_test, t:t_test}))
                plotAccuracyTrainUser.append(sess.run(accuracy, feed_dict={X:X_batch,t: t_batch}))
                plotAccuracyTestUser.append(sess.run(accuracy, feed_dict={X:X_test,t: t_test}))

                # save model
                #saver.save(sess, "save/model.ckpt", global_step=global_step)
                # stock value at the last epoch (for the graph)
            """if(step==0):
                plotLossTrainUser.append(sess.run(score, feed_dict={X:train_input, t:train_output}))
                plotLossTestUser.append(sess.run(score, feed_dict={X:test_input, t:test_output}))
                plotAccuracyTrainUser.append(sess.run(accuracy, feed_dict={X: train_input,t: train_output}))
                plotAccuracyTestUser.append(sess.run(accuracy, feed_dict={X: test_input,t: test_output}))"""

        print("Optimization Finished!")

        # final accuracy
        #print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_input, t: test_output}))

    return plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser



timestep= 1
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
    restore_model = False
    learning_rate = 0.0001
    batch_size = 20
    display_step = 20000
    num_epoch = 1000
    display_epoch = 100
    train_size = 0.7
    num_user = 1 # number of user to procces

    # Network Parameters
    n_hidden_1 = 20 # 1st layer number of neurons
    n_hidden_2 = 20 # 2nd layer number of neurons
    num_input = numberOfPOIs*j # same vector for all user based and all POIs
    num_classes = numberOfPOIs # same vector for all user based and all POIs

    # for each user wanted we procced a linear model and stock loss and accuracy
    for i in range(0,num_user):
        plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser = runLSTM(i, input[i], output[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hidden_1, num_input, num_classes, train_size, j)
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
