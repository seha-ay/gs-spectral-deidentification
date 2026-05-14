import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
import math
import gc
from collections import Counter
import sys
import time
import sys


#@title GS
import cupy as cp
import scipy.linalg
from scipy.linalg import dft

def dft_matrix(n):
    m = dft(n, scale='sqrtn')
    return m

def imaginer_cp(t):
    return cp.exp(1j * t)

def gs3D_GPU(data, iter, maskP=0):
    data = cp.array(data)
    
    FF1 = cp.array(dft_matrix(data.shape[1]))
    FF2 = cp.array(dft_matrix(data.shape[2]))
    invFF1 = cp.linalg.inv(FF1)
    invFF2 = cp.linalg.inv(FF2)
    FF_tensor1 = cp.tile(FF1,(len(data),1,1))
    invFF_tensor1 = cp.tile(invFF1,(len(data),1,1))
    FF_tensor2 = cp.tile(FF2,(len(data),1,1))
    invFF_tensor2 = cp.tile(invFF2,(len(data),1,1))
    

    random_matrix = cp.random.uniform(low=0, high=2*cp.pi, size=(data.shape))
    vfunc = cp.vectorize(imaginer_cp)
    random_matrix_2 = vfunc(random_matrix)
    
    for i in range(iter):
        mask = cp.random.choice((0,1), (data.shape), p = [maskP, 1-maskP])
        transformed = FF_tensor1@(data * random_matrix_2)@FF_tensor2
        mag_transformed = transformed/cp.abs(transformed)
        back_transformed = invFF_tensor1@(mag_transformed*mask)@invFF_tensor2
        
        angles = cp.angle(back_transformed)
        random_matrix_2 = vfunc(angles)
    
    ans = cp.abs(back_transformed)
    return ans.get()


#@title GS Batch
def GS_batch_image(data, batch_size, ite, maskP=0):
    '''
    This function divides data into batches to iterate within the GS,
    ensuring efficient RAM and GPU usage. Handles cases where the last batch
    contains fewer samples than the batch size by processing it separately.
    '''
    # Create an empty array to hold the data
    gs_array = np.empty((0, data.shape[1], data.shape[2]))
    
    # Calculate the total number of full batches
    n_batch = len(data) // batch_size
    
    # Batch iteration starts here
    for i in range(0, n_batch):
        # Process a full batch
        gs_batch = gs3D_GPU(data[i * batch_size:(i + 1) * batch_size], ite, maskP=maskP)
        gs_array = np.append(gs_array, gs_batch, axis=0)
        sys.stdout.write(f"\rBatch {i + 1} of {n_batch} (full batch) completed...")
        sys.stdout.flush()
        time.sleep(0.001)
        del gs_batch
        gc.collect()
    
    # Handle the remaining data as a smaller batch
    if len(data) % batch_size != 0:
        remaining_data = data[n_batch * batch_size:]
        gs_batch = gs3D_GPU(remaining_data, ite, maskP=maskP)
        gs_array = np.append(gs_array, gs_batch, axis=0)
        sys.stdout.write(f"\nRemaining data batch completed (size: {len(remaining_data)})...")
        sys.stdout.flush()
        time.sleep(0.001)
        del gs_batch
        gc.collect()
    
    return gs_array
