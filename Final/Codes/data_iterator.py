import numpy as np

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

# split data by batch for each epoch and shuffle if asked
def simple_data_iterator2(X, t, num_timesteps):

    def stack_timesteps_block(X, t, num_timesteps, stride=1):
        X_stacked = np.stack([X[k:k+num_timesteps,:] for k in range(0, X.shape[0]-num_timesteps+1, stride)], axis=0)
        t_stacked = t[num_timesteps-1::stride]
        return X_stacked, t_stacked

    X_stacked, t_stacked = stack_timesteps_block(X, t, num_timesteps)

    return X_stacked, t_stacked

# split data by batch for each epoch and shuffle if asked
def simple_data_iterator3(X, t, batch_size, num_epochs, shuffle):
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