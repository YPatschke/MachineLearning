from __future__ import print_function

import numpy as np
import tensorflow as tf

import loadData as ld

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

# get occurances per index POIs
def getOccurances(input, output):
    occurances=[]
    occ=[]
    occ2=[]
    for k in range(0,238):
        occurances.append(sum(sum(output)))
        occ.append(sum(output)[k])
        occ2.append((occurances[k]-occ[k])/(occ[k]+1e-6))

    return occ2

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

# LSTM model run
def runLSTM(user, input, output, learning_rate, batch_size, display_step, num_epoch, display_epoch, n_hiddens, n_neurons, num_input, num_classes, train_size, timestep):
    # clears the default graph stack and resets the global default graph
    tf.reset_default_graph()

    # split data and get all input, output and number of example uses in the train
    train_input, train_output, test_input, test_output, num_examples = split(input, output, train_size, False)

    # get occurences of POIs
    occurences = getOccurances(input,output)

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

    # define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    """weights = tf.constant(occurences)
    print("weights:"+str(weights))
    loss_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=prediction, targets=Y, pos_weight=weights))"""
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
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
        train_batch_gen = simple_data_iterator(train_input, train_output, batch_size, num_epoch, timestep, True)

        train_input, train_output = simple_data_iterator2(train_input, train_output, timestep)
        test_input, test_output = simple_data_iterator2(test_input, test_output, timestep)

        # variable to store my results
        LossTrainUser =[]
        LossTestUser=[]
        AccuracyTrainUser=[]
        AccuracyTestUser=[]
        TrueMovement=[]
        PredictionMovement=[]

        # begin to learn by processing over my training data
        for batch_x, batch_y, epoch, step in train_batch_gen:
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            # display informations sometimes
            if((epoch % display_epoch == 0 or epoch == 1) and (step % display_step == 0 or step == num_steps-1)):
                # calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print("Hidden: "+str(n_hiddens)+", Neurons: "+str(n_neurons)+", Timestep: "+str(timestep)+", User: "+str(user)+",Epoch: "+str(epoch)+", Step "+str(step)+", Minibatch Loss= "+str(loss)+", Training Accuracy= "+str(acc))
                cost_train, accurarcy_train = sess.run([loss_op, accuracy], feed_dict={X: train_input, Y: train_output})
                cost_test, accurarcy_test = sess.run([loss_op, accuracy], feed_dict={X: test_input, Y: test_output})
                pred_test_batch = sess.run(prediction, feed_dict={X:batch_x})
                pred_test = sess.run(prediction, feed_dict={X:test_input})
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
            if step == 0:
                cost_train, accurarcy_train = sess.run([loss_op, accuracy], feed_dict={X: train_input, Y: train_output})
                cost_test, accurarcy_test, pred_test = sess.run([loss_op, accuracy, prediction], feed_dict={X: test_input, Y: test_output})
                LossTrainUser.append(cost_train)
                LossTestUser.append(cost_test)
                AccuracyTrainUser.append(accurarcy_train)
                AccuracyTestUser.append(accurarcy_test)
                TrueMovement=np.argmax(test_output, axis=1)
                PredictionMovement=np.argmax(pred_test, axis=1)

            if(step == num_steps-1 and epoch == num_epoch-1):
                # save model
                saver.save(sess, "C:/Users/Yannick/Desktop/Image1/LSTM/Save/"+"U"+str(user)+"H"+str(n_hiddens)+"N"+str(n_neurons)+"T"+str(timestep)+"Model.ckpt")

    print("Optimization Finished!")

    return LossTrainUser, LossTestUser, AccuracyTrainUser, AccuracyTestUser, TrueMovement, PredictionMovement

# Parameters
n_hiddens = [1,5] # number of hidden layers
n_neurons = [1,10] # number of neurons per hidden layers
timesteps= [1,2] # number of timestep, so number of previous result took into account
learning_rate = 0.001 # learning rate (how fast thee model choose to correcte/learn)
batch_size = 20 # number of example we train in same times
display_step = 2000 # if we want to display more often information during training
num_epoch = 1000 # number of time that we prcess all the data in trainning
display_epoch = 500 # if we want to display more often information during training
train_size = 0.7 # part size of data that will be use for training
num_user = 2 # number of user to procces (in our case all user 150 if we want a model per users)

plotLossTrain=[]
plotLossTest=[]
plotAccuracyTrain=[]
plotAccuracyTest=[]

# get data
input, output, number, numberOfPOIs = ld.get_data(True,1)

plotTrueMovement=[]
plotPredictionMovement=[]
for hidden in n_hiddens:
    plotLossTrainHidden=[]
    plotLossTestHidden=[]
    plotAccuracyTrainHidden=[]
    plotAccuracyTestHidden=[]

    plotTrueMovementHidden=[]
    plotPredictionMovementHidden=[]
    for neuron in n_neurons:
        plotLossTrainNeuron=[]
        plotLossTestNeuron=[]
        plotAccuracyTrainNeuron=[]
        plotAccuracyTestNeuron=[]

        plotTrueMovementNeuron=[]
        plotPredictionMovementNeuron=[]
        for timestep in timesteps:

            # variables for stock loss and accuracy at different epoch
            plotLossTrainTimeStep=[]
            plotLossTestTimestep=[]
            plotAccuracyTrainTimestep=[]
            plotAccuracyTestTimestep=[]

            plotTrueMovementTimestep=[]
            plotPredictionMovementTimestep=[]

            # Network Parameters
            num_input = numberOfPOIs # same vector for all user based and all POIs
            num_classes = numberOfPOIs # same vector for all user based and all POIs

            # for each user wanted we procced a linear model and stock loss and accuracy
            for i in range(0,num_user):
                plotLossTrainUser, plotLossTestUser, plotAccuracyTrainUser, plotAccuracyTestUser, plotTrueMovementUser, plotPredictionMovementUser = runLSTM(i, input[i], output[i], learning_rate, batch_size, display_step, num_epoch, display_epoch, hidden, neuron, num_input, num_classes, train_size, timestep)
                plotLossTrainTimeStep.append(plotLossTrainUser)
                plotLossTestTimestep.append(plotLossTestUser)
                plotAccuracyTrainTimestep.append(plotAccuracyTrainUser)
                plotAccuracyTestTimestep.append(plotAccuracyTestUser)
                plotTrueMovementTimestep.append(plotTrueMovementUser)
                plotPredictionMovementTimestep.append(plotPredictionMovementUser)

            plotLossTrainNeuron.append(plotLossTrainTimeStep)
            plotLossTestNeuron.append(plotLossTestTimestep)
            plotAccuracyTrainNeuron.append(plotAccuracyTrainTimestep)
            plotAccuracyTestNeuron.append(plotAccuracyTestTimestep)
            plotTrueMovementNeuron.append(plotTrueMovementTimestep)
            plotPredictionMovementNeuron.append(plotPredictionMovementTimestep)

        plotLossTrainHidden.append(plotLossTrainNeuron)
        plotLossTestHidden.append(plotLossTestNeuron)
        plotAccuracyTrainHidden.append(plotAccuracyTrainNeuron)
        plotAccuracyTestHidden.append(plotAccuracyTestNeuron)
        plotTrueMovementHidden.append(plotTrueMovementNeuron)
        plotPredictionMovementHidden.append(plotPredictionMovementNeuron)

    plotLossTrain.append(plotLossTrainHidden)
    plotLossTest.append(plotLossTestHidden)
    plotAccuracyTrain.append(plotAccuracyTrainHidden)
    plotAccuracyTest.append(plotAccuracyTestHidden)
    plotTrueMovement.append(plotTrueMovementHidden)
    plotPredictionMovement.append(plotPredictionMovementHidden)

plotLossTrain=np.array(plotLossTrain)
plotLossTest=np.array(plotLossTest)
plotAccuracyTrain=np.array(plotAccuracyTrain)
plotAccuracyTest=np.array(plotAccuracyTest)
plotTrueMovement=np.array(plotTrueMovement)
plotPredictionMovement=np.array(plotPredictionMovement)


a = np.array(plotLossTrain)
np.save("C:/Users/Yannick/Desktop/Image1/LSTM/Data/LSTMPlotLossTrain.npy", a)
a = np.array(plotLossTest)
np.save("C:/Users/Yannick/Desktop/Image1/LSTM/Data/LSTMPlotLossTest.npy", a)
a = np.array(plotAccuracyTrain)
np.save("C:/Users/Yannick/Desktop/Image1/LSTM/Data/LSTMPlotAccuracyTrain.npy", a)
a = np.array(plotAccuracyTest)
np.save("C:/Users/Yannick/Desktop/Image1/LSTM/Data/LSTMPlotAccuracyTest.npy", a)
a = np.array(plotTrueMovement)
np.save("C:/Users/Yannick/Desktop/Image1/LSTM/Data/LSTMPlotTrue.npy", a)
a = np.array(plotPredictionMovement)
np.save("C:/Users/Yannick/Desktop/Image1/LSTM/Data/LSTMPlotPrediction.npy", a)