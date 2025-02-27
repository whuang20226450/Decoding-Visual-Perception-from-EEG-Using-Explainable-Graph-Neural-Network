import scipy.io as sio
import numpy as np
import random

for i in range(1,11):
    print(f'Processing S{i}...')
    
    data_in = sio.loadmat(f'S{i}.mat')
    x = data_in['X_2D']
    y = data_in['categoryLabels'].reshape(-1,1)

    x, y = np.array(x), np.array(y)

    np.random.seed(0)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x_shuffled, y_shuffled = x[indices], y[indices]

    data = {
        'x': x,
        'y': y
    }
    np.save(f'../processed_v2/S{i}.npy', data)

    print(f'S{i} done!')
    print(x.shape, y.shape)
    print('-------------------\n')