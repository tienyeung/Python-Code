import numpy as np
import random
from multiprocessing import Queue
#from scipy.sparse import lil_matrix, csr_matrix, dok_matrix


class PairSampler(object):

    def __init__(self, matrix, batch_size):
        self.result_queue = Queue()
        self.batch_size = batch_size
        matrix=matrix.tocsr()
        self.rows = matrix.nonzero()[0]
        self.cols = matrix.nonzero()[1]
        self.vals = matrix.data

    def generate_batches(self, shuffle=True): 
        if shuffle is True:
            rng_state = np.random.get_state()
            np.random.shuffle(self.rows)
            np.random.set_state(rng_state)
            np.random.shuffle(self.cols)
            np.random.set_state(rng_state)
            np.random.shuffle(self.vals)
        
        for i in range(int(len(self.vals) / self.batch_size) + 1):        
            self.result_queue.put(i)       
    
    def is_empty(self):
        return self.result_queue.empty()
    
    def next_batch(self):
        i= self.result_queue.get()   
        if (i + 1) * self.batch_size < len(self.vals):
                batch_row = self.rows[i * self.batch_size: (i + 1) * self.batch_size]
                batch_col = self.cols[i * self.batch_size: (i + 1) * self.batch_size]
                batch_val = self.vals[i * self.batch_size: (i + 1) * self.batch_size]
        else:
                batch_row = self.rows[i * self.batch_size:]
                batch_col = self.cols[i * self.batch_size:]
                batch_val = self.vals[i * self.batch_size:]
            
        return batch_row.tolist(), batch_col.tolist(), batch_val.tolist()
