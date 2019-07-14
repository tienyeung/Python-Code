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



class DMF():

    def __init__(self, n_users, n_items,  layers=[64,32,16,8], learning_rate=0.001, keep_prob=[0.8,0.8,0.8,0.8], reg_param=0, random_seed=2018):
        # bind params to class
        self.layers = layers  # perception_layers
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.learning_rate = learning_rate
        self.n_users = n_users
        self.n_items = n_items
        self.reg_param = reg_param
        # init all variables in a tensorflow graph
        np.random.seed(2018)
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
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.users = tf.placeholder(tf.int32, shape=[None])  # None * 1
            self.items = tf.placeholder(tf.int32, shape=[None])  # None * 1
            self.ratings = tf.placeholder(tf.float32, shape=[None])  # None * 1

            # Variables.
            self.weights = self._initialize_weights()
            
            # _________ Embedding Layer _____________
            self.u_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.users)  # None * 1* embedding_size
            self.i_emb = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.items)  # None * 1* embedding_size

            
            # ________ Perception Layers __________
            for i in range(1, len(self.layers)):
                self.u_emb = tf.add(tf.matmul(self.u_emb, self.weights['layer_user_%d' % i]), self.weights['bias_user_%d' % i])  # None * layer[i] * 1
                self.u_emb = tf.nn.dropout(tf.nn.relu(self.u_emb), self.dropout_keep[i])  # dropout at each Deep layer
                
                self.i_emb = tf.add(tf.matmul(self.i_emb, self.weights['layer_item_%d' % i]), self.weights['bias_item_%d' % i])  # None * layer[i] * 1
                self.i_emb = tf.nn.dropout(tf.nn.relu(self.i_emb), self.dropout_keep[i])  # dropout at each Deep layer


            ## ________ Combination Layer __________ 
            self.combination=tf.multiply(self.u_emb, self.i_emb)

            # ________ Prediction Layer __________ 
            self.out = tf.reduce_sum(tf.matmul(self.combination, self.weights['prediction']), 1, keepdims=True)  # None * 1
            
            
            # Compute the loss.
            self.loss = tf.nn.l2_loss(tf.subtract(self.out, tf.expand_dims(self.ratings,1)))
            
            if self.reg_param > 0:
                regularizer = tf.contrib.layers.l2_regularizer(self.reg_param)(self.u_emb) + tf.contrib.layers.l2_regularizer(self.reg_param)(self.i_emb)
                for i in range(1, len(self.layers)):
                    regularizer = regularizer+ tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['layer_user_%d' % i]) +\
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['bias_user_%d' % i])+\
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['layer_item_%d' % i]) +\
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['bias_item_%d' % i])+\
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['layer_combination'])+\
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['bias_combination'])+\
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['prediction'])
                                
                self.loss = self.loss + regularizer
                
            # Optimizer.
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()
            
    def _initialize_weights(self):
        all_weights = dict()
        # _________ Embedding Layer _____________
        all_weights['user_embeddings'] = tf.Variable(
            tf.random_normal([self.n_users, self.layers[0]], 0.0, 0.01), name='user_embeddings')  # features_U* K
        all_weights['item_embeddings'] = tf.Variable(
            tf.random_normal([self.n_items, self.layers[0]], 0.0, 0.01), name='item_embeddings')  # features_I * K     
        
        # ________ Perception Layers __________
        if len(self.layers) > 0:
            for i in range(1, len(self.layers)):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_user_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_user_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
            for i in range(1, len(self.layers)):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_item_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_item_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
            
            all_weights['layer_combination']=tf.Variable(tf.random_normal([self.layers[-1], self.layers[-1]], -1.0, 1.0))  # features_I * K
            all_weights['bias_combination']=tf.Variable(tf.random_normal([self.layers[-1], self.layers[-1]], -1.0, 1.0))  # features_I * K
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)), dtype=np.float32)  # layers[-1] * 1
        
        return all_weights

    def train(self, log, sampler, train, test, iters):
        
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
                _, batch_loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.users: users, self.items: items, self.ratings:ratings, self.dropout_keep:self.keep_prob})
                loss += batch_loss
                
            t2 = time()
            print("Training: %.2fs loss: %f " % (t2 - t1, loss))
            log.write("Training: %.2fs loss: %f " % (t2 - t1, loss))
            
            if (i + 1) % 1 == 0:
                mae, mse, rmse = eval.evalRatingPerformance(self, self.no_dropout)
                t3 = time()
                print("Test: %fs MAE: %f MSE: %f RMSE: %f" % (t3 - t2, mae, mse, rmse))
                log.write("Test:%fs MAE: %f MSE: %f RMSE: %f \n" % (t3 - t2, mae, mse, rmse))

            log.flush()


if __name__ == '__main__':
    BATCH_SIZE = [64, 64, 64, 64, 64]
    MAX_ITERS = [30, 20, 20, 100, 100]
    DATASET_NAMES = ['ml-100k', 'lastfm', 'ml-1m', 'kindle-store', 'epinion']
    layers=[64,32,16,8]
    keep_prob=[0.8,0.8,0.8,0.8]
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.txt', binary=False)
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_ratio(split_ratio=(8, 2))
        sampler = PairSampler(train, BATCH_SIZE[i])   
        
        log = open('../log/' + DATASET_NAMES[i] + '.DMF.log', 'a') 
        log.write("############################################### \n")
        log.write("batch_size=%d \n" % (BATCH_SIZE[i]))
        log.write("layers=%s, keep_prob=%s\n" % (layers,keep_prob))
        log.flush()      

        model = DMF(n_users=n_users, n_items=n_items, layers=layers, keep_prob=keep_prob)
        model.train(log, sampler, train, test, MAX_ITERS[i])
        log.close()

