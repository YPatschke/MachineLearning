import lstm
import time
import matplotlib.pyplot as plt
import numpy as np

def plot_results(predicted_data, true_data, user, timestep):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    #plt.show()
    plt.title('User: '+str(user)+' Timestep: '+str(timestep)+' - True data vs Prediction (point by point)')
    plt.savefig('C:/Users/Yannick/Desktop/Image1/LSTM2/user'+str(user)+'Timestep'+str(timestep)+'Graph.png')
    plt.close(fig)

def plot_results_multiple(predicted_data, true_data, prediction_len, user, timestep):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    #plt.show()
    plt.title('User: '+str(user)+' Timestep: '+str(timestep)+' - True data vs Prediction (next seq)')
    plt.savefig('C:/Users/Yannick/Desktop/Image1/LSTM2/user'+str(user)+'Timestep'+str(timestep)+'GraphMultiple.png')
    plt.close(fig)

seq_len = 5
num_users=1
for j in range(1, seq_len+1):
	for user in range(0, num_users):
		#Main Run Thread
		if __name__=='__main__':
			global_start_time = time.time()
			epochs  = 100
			#seq_len = 5

			# to predict
			seqPred=10

			print('> Loading data... ')

			X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', j, True, user)

			print('> Data Loaded. Compiling...')

			model = lstm.build_model([1, j, 100, 1]) # [1, 50, 100, 1]

			model.fit(
				X_train,
				y_train,
				batch_size=20, #512
				nb_epoch=epochs,
				validation_split=0.05)

			predictions = lstm.predict_sequences_multiple(model, X_test, j, seqPred) #50
			#predicted = lstm.predict_sequence_full(model, X_test, seq_len)
			predicted2 = lstm.predict_point_by_point(model, X_test)

			print('Training duration (s) : ', time.time() - global_start_time)
			plot_results_multiple(predictions, y_test, seqPred, user, j) #50

			#plot_results(predicted, y_test)
			plot_results(predicted2, y_test, user, j)

			predicted2=np.array([int(i) for i in predicted2])
			print(predicted2.shape)
			print(predicted2)
			print(y_test)
			accuracy=0
			for i in range(0, len(predicted2)):
				if predicted2[i] == y_test[i]:
					accuracy+=1
			accuracy=accuracy/len(predicted2)
			print(accuracy)