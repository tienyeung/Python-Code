import sys 
sys.path.append("..") 
import functools
import numpy
import tensorflow as tf
from time import time
import numpy as np
import toolz
from tqdm import tqdm
from evaluator import Evaluator
from sampler import PairSampler
from datatool import DataProcessor

                        
class SVD:
    '''
    Koren Y, Bell R, Volinsky C. Matrix factorization techniques for recommender systems[J]. Computer, 2009 (8): 30-37.
    '''
    
    def __init__(self, n_users, n_items, mu, embed_dim=50, learning_rate=0.001, reg_param=0.001, random_seed=2018):
        self.n_users = n_users
        self.n_items = n_items
        self.mu=mu
        self.embed_dim = embed_dim
        self.reg_param = reg_param
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            
            # Input data.
            self.users = tf.placeholder(tf.int32, shape=[None])  # None
            self.items = tf.placeholder(tf.int32, shape=[None])  # None
            self.ratings = tf.placeholder(tf.float32, shape=[None])  # None * 1
            mu = tf.constant([self.mu], dtype=tf.float32)
            
            # Variables.
            self.weights = { 
                'user_embeddings':tf.Variable(tf.random_normal([self.n_users, self.embed_dim], 0.0, 0.01)),
                'item_embeddings':tf.Variable(tf.random_normal([self.n_items, self.embed_dim], 0.0, 0.01)),
                'user_biases':tf.Variable(tf.tf.zeros([self.n_users, 1], 0.0, 0.01)),
                'item_biases':tf.Variable(tf.random_normal([self.n_items, 1], 0.0, 0.01)),
                }
            
            self.u_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.users)  # None * 1* embed_dim
            self.i_emb = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.items)  # None * 1* embed_dim
            
            self.u_bias = tf.nn.embedding_lookup(self.weights['user_biases'], self.users)  # None * 1* embed_dim
            self.i_bias = tf.nn.embedding_lookup(self.weights['item_biases'], self.items)  # None * 1* embed_dim
            
            # prediction    
            self.out = tf.add(mu, self.u_bias + self.i_bias + tf.reduce_sum(tf.multiply(self.u_emb, self.i_emb), 1, keepdims=True))
            
            # regularization
            regularizer = tf.contrib.layers.l2_regularizer(self.reg_param)(self.u_emb) + \
                          tf.contrib.layers.l2_regularizer(self.reg_param)(self.i_emb) + \
                          tf.contrib.layers.l2_regularizer(self.reg_param)(self.u_bias) + \
                          tf.contrib.layers.l2_regularizer(self.reg_param)(self.i_bias) 
            
            # compute the loss.
            self.loss = tf.nn.l2_loss(tf.subtract(self.out, tf.expand_dims(self.ratings, 1))) + regularizer
            
            # optimizer.
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)     
                    
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()

    def train(self, log, sampler, train, test, iters):
        
        log.write("embed_dim=%d \n" % (self.embed_dim))
        log.write("############################################### \n")
        
        eval = Evaluator(test)
        for i in range(iters):
            print("Iter %d..." % (i + 1))
            log.write("Iter %d...\t" % (i + 1))
            
            loss = 0
            t1 = time()
            sampler.generate_batches()
            while not sampler.is_empty():
                users, items, ratings = sampler.next_batch()
                _, batch_loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.users: users, self.items: items, self.ratings:ratings})
                loss += batch_loss
                
            t2 = time()
            print("Training: %.2fs loss: %f " % (t2 - t1, loss))
            log.write("Training: %.2fs loss: %f " % (t2 - t1, loss))
            
            if (i + 1) % 1 == 0:
                mae, mse, rmse = eval.evalRatingPerformance(self)
                t3 = time()
                print("Test: %fs MAE: %f MSE: %f RMSE: %f" % (t3 - t2, mae, mse, rmse))
                log.write("Test:%fs MAE: %f MSE: %f RMSE: %f \n" % (t3 - t2, mae, mse, rmse))

            log.flush()


if __name__ == '__main__':
    BATCH_SIZE = [64, 64, 64, 64, 64]
    MAX_ITERS = [30, 20, 20, 100, 100]
    DATASET_NAMES = ['ml-100k', 'lastfm', 'ml-1m', 'kindle-store', 'epinion']
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.txt', binary=False)
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_ratio(split_ratio=(8, 2))
        sampler = PairSampler(train, BATCH_SIZE[i])   
        
        log = open('../log/' + DATASET_NAMES[i] + '.SVD.log', 'a') 
        log.write("############################################### \n")
        log.write("batch_size=%d \n" % (BATCH_SIZE[i]))
        log.flush()      
        for embed_dim in [8, 16, 32, 64]:
            model = SVD(n_users=n_users, n_items=n_items, mu=train.mean(), embed_dim=embed_dim, learning_rate=0.001)
            model.train(log, sampler, train, test, MAX_ITERS[i])
        log.close()
