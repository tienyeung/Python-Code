import argparse, math
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error

from DataTool import DataLoader
import numpy as np
import tensorflow as tf


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run ACF_SE.")
    parser.add_argument('--embedding_size', type=int, default=64, help='Number of hidden factors.')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='GradientDescentOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')

    #    parser.add_argument('--early_stop', type=int, default=1, help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()


class ACF_SE():

    def __init__(self, num_users, num_items, num_neighbors, n_neg_samples, reg_param, embedding_size, epoch, t1_batch_size, t2_batch_size, learning_rate, optimizer_type, verbose, random_seed=2018):
        # bind params to class
        self.t1_batch_size = t1_batch_size
        self.t2_batch_size = t2_batch_size
        self.embedding_size = embedding_size  #
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.num_users = num_users
        self.num_items = num_items
        self.num_neighbors = num_neighbors
        self.n_neg_samples = n_neg_samples
        self.reg_param = reg_param
        self.emb_list = []
        #        self.early_stop = early_stop
        # performance of each epoch
        self.train_rmse, self.test_rmse = [], []
        # init all variables in a tensorflow graph
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
            self.users = tf.placeholder(tf.int32, shape=[None, 1])  # None * 1
            self.items = tf.placeholder(tf.int32, shape=[None, 1])  # None * 1
            self.ratings = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.neighbors = tf.placeholder(tf.int32, shape=[None, None])  # None *num_neighbors
            
            # Variables.
            self.weights = self._initialize_weights()

            #####################################################
            #     Attention Model for rating prediction         #    
            #####################################################
            
            # _________ Embedding Layer _____________
            self.u_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.users)  # None * 1* embedding_size
            self.i_emb = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.items)  # None * 1* embedding_size
            self.n_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.neighbors)  # None * num_neighbors* embedding_size
            
            self.u_emb = tf.reshape(self.u_emb, shape=[-1, self.embedding_size])
            self.i_emb = tf.reshape(self.i_emb, shape=[-1, self.embedding_size])
            
            #self.emb_list.append(self.u_emb)
            for i in range(0, self.num_neighbors): 
                self.emb_list.append(self.n_emb[:, i, :])
                
            # ________ Attention Layers __________
            self.emb_list = tf.stack(self.emb_list)  # (num_friends)* None * K
            self.emb_list = tf.transpose(self.emb_list, perm=[1, 0, 2])  # None * (num_friend)* K
    
            self.Wn = tf.matmul(tf.reshape(self.emb_list, shape=[-1, self.embedding_size]), self.weights['attention_W'])  # None*(num_neighbors+1)*embedding_size
            self.Wn = tf.reshape(self.Wn, shape=[-1, self.num_neighbors, self.embedding_size])
 
            self.Xu = tf.matmul(self.u_emb, self.weights['attention_X'])  # None*1*(num_neighbors+1)
            self.Xu = tf.reshape(self.Xu, shape=[-1, 1, self.embedding_size])
            
            attention_factors = self.Wn + self.Xu + self.weights['attention_b']
            
            self.attention_out = tf.reduce_sum(tf.multiply(self.weights['attention_h'], tf.nn.relu(attention_factors)), 2, keepdims=True) 
            self.attention_exp = tf.exp(self.attention_out)  
            self.attention_sum = tf.reduce_sum(self.attention_exp, 1, keepdims=True)
            self.attention_out = tf.div(self.attention_exp, self.attention_sum)

            trust_emb = tf.reduce_sum(tf.multiply(self.attention_out, self.emb_list), 1)  # None *K
            trust_emb =tf.add(trust_emb, self.u_emb)
            
            # ________ Prediction Layer __________
            self.predictions = tf.reduce_sum(tf.multiply(trust_emb, self.i_emb), 1, keepdims=True)
            
            # Compute the loss.
            self.t1_loss = tf.nn.l2_loss(tf.subtract(self.ratings, self.predictions))
            if self.reg_param > 0:
                regularizer = tf.contrib.layers.l2_regularizer(self.reg_param)(self.u_emb) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.i_emb) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.n_emb) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['attention_W']) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['attention_X']) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['attention_h']) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param)(self.weights['attention_b'])
                                
                self.t1_loss = self.t1_loss + regularizer

            ###########################################
            #     Model for social network embedding  #    
            ###########################################
            
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.nodes = tf.placeholder(tf.int32, shape=[None])  # None * 1
            
            self.node_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.nodes)  # batch_size * embedding_size

            # Compute the loss of social embedding model, using a sample of the negative labels each time.
            self.t2_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.weights['out_embeddings'], self.weights['biases'],
                                           self.labels, self.node_emb, self.n_neg_samples, self.num_users))
            
            # Optimizer.
            
            if self.optimizer_type == 'AdamOptimizer':
                self.t1_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.t1_loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.t1_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.t1_loss)

            self.t2_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.t2_loss)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['user_embeddings'] = tf.Variable(tf.random_uniform([self.num_users, self.embedding_size], -1.0, 1.0))  # features_U* K
        all_weights['item_embeddings'] = tf.Variable(tf.random_uniform([self.num_items, self.embedding_size], -1.0, 1.0))  # features_I * K
        
        # _________ Attention Layer _____________
        all_weights['attention_W'] = tf.Variable(tf.random_uniform([self.embedding_size, self.embedding_size], -1.0, 1.0)) 
        all_weights['attention_X'] = tf.Variable(tf.random_uniform([self.embedding_size, self.embedding_size], -1.0, 1.0)) 
        
        all_weights['attention_b'] = tf.Variable(tf.random_uniform([1, self.embedding_size], -1.0, 1.0))
        all_weights['attention_h'] = tf.Variable(tf.random_uniform([self.embedding_size], -1.0, 1.0))
             
        # _________ Social Embedding Layer _____________
        all_weights['out_embeddings'] = tf.Variable(tf.truncated_normal([self.num_users, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        all_weights['biases'] = tf.Variable(tf.zeros([self.num_users]))
        
        return all_weights

    def t1_partial_fit(self, data):  # fit a batch
        feed_dict = {self.users: data['U'], self.items: data['I'], self.neighbors: data['N'], self.ratings: data['R']}
        loss, opt = self.sess.run((self.t1_loss, self.t1_optimizer), feed_dict=feed_dict)
        return loss

    def t2_partial_fit(self, data):  # fit a batch
        feed_dict = {self.nodes: data['U'], self.labels: data['V']}
        loss, opt = self.sess.run((self.t2_loss, self.t2_optimizer), feed_dict=feed_dict)
        return loss

    @staticmethod
    def get_random_block_from_t1_data(data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['R']) - batch_size)
        U, I, R, N = [], [], [], []
        # forward get sample
        i = start_index
        while len(U) < batch_size and i < len(data['U']):
            R.append([data['R'][i]])
            U.append([data['U'][i]])
            I.append([data['I'][i]])
            N.append(data['N'][i])
            i = i + 1
        # backward get sample
        i = start_index
        while len(U) < batch_size and i >= 0:
            R.append([data['R'][i]])
            U.append([data['U'][i]])
            I.append([data['I'][i]])
            N.append(data['N'][i])
            i = i - 1
        return {'U': U, 'I': I, 'R': R, 'N':N}

    @staticmethod
    def get_random_block_from_t2_data(data, batch_size):  # generate a random block of training data
        batch_xs = {}
        start_index = np.random.randint(0, len(data['U']) - batch_size)
        batch_xs['U'] = data['U'][start_index:(start_index + batch_size)]
        # print(data['U'][start_index:(start_index + batch_size)])
        batch_xs['V'] = data['V'][start_index:(start_index + batch_size)]
        # print(data['V'][start_index:(start_index + batch_size)])
        return batch_xs

    @staticmethod
    def shuffle_in_unison_scary(a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
    
    @staticmethod
    def shuffle_in_unison_scary1(a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def pre_train_embedding(self, Link_data, total_epochs):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            # Write configurations in log
            f1 = open('dmfse.log', 'a')
            f1.write("### ACF_SE ###\n")
            t2 = time()
            
            for epoch in range(total_epochs):
                t1 = time()
                # learning social network embedding
                total_batch = int(len(Link_data['U']) / self.t2_batch_size)
                
                rng_state = np.random.get_state()
                np.random.shuffle(Link_data['U'])
                np.random.set_state(rng_state)
                np.random.shuffle(Link_data['V'])
                t2_loss = 0
                for i in range(total_batch):
                    batch_xs = self.get_random_block_from_t2_data(Link_data, self.t2_batch_size)
                    t2_loss = t2_loss + self.t2_partial_fit(batch_xs)
                t2 = time()
                                         
                if self.verbose > 0 and epoch % self.verbose == 0:
                    print("Epoch %d: [%.1f s]\t loss=%.4f [%.1f s]" % (epoch + 1, t2 - t1, t2_loss, time() - t2))
                    f1.write("Epoch %d: [%.1f s]\t loss=%.4f [%.1f s]\n" % (epoch + 1, t2 - t1, t2_loss, time() - t2))
                f1.flush()    
            f1.close()
            
    def train(self, Train_data, Test_data, Link_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            # Write configurations in log
            f1 = open('dmfse.log', 'a')
            f1.write("embedding_size=%d \n" % (self.embedding_size))
            t2 = time()
            # init_train = self.evaluate(Train_data)
            # init_test = self.evaluate(Test_data)
            # print("Initialization: \t train=[(%.4f %.4f)], test=[(%.4f %.4f)]  [%.1f s]" % (init_train[0], init_train[1], init_test[0], init_test[1], time() - t2))
            # f1.write("Initialization: \t train=[(%.4f %.4f)], test=[(%.4f %.4f)]  [%.1f s]\n" % (init_train[0], init_train[1], init_test[0], init_test[1], time() - t2))

            for epoch in range(self.epoch):
                t1 = time()
                # training deep matrix factorization model
                self.shuffle_in_unison_scary(Train_data['U'], Train_data['I'], Train_data['N'], Train_data['R'])
                total_batch = int(len(Train_data['R']) / self.t1_batch_size)
                t1_loss = 0
                for i in range(total_batch):
                    batch_xs = self.get_random_block_from_t1_data(Train_data, self.t1_batch_size)
                    t1_loss = t1_loss + self.t1_partial_fit(batch_xs)
               
                # learning social network embedding
                t2_loss = 0
                for k in range(0, 1):
                    rng_state = np.random.get_state()
                    np.random.shuffle(Link_data['U'])
                    np.random.set_state(rng_state)
                    np.random.shuffle(Link_data['V'])
                    total_batch = int(len(Link_data['U']) / self.t2_batch_size)
                    t2_loss = 0
                    for i in range(total_batch):
                        batch_xs = self.get_random_block_from_t2_data(Link_data, self.t2_batch_size)
                        t2_loss = t2_loss + self.t2_partial_fit(batch_xs)
                t2 = time()

                train_result = self.evaluate(Train_data)
                test_result = self.evaluate(Test_data)

                # self.train_rmse.append(train_result)
                # self.valid_rmse.append(valid_result)
                self.test_rmse.append(test_result)
                         
                if self.verbose > 0 and epoch % self.verbose == 0:
                    # print("Epoch %d: [%.1f s]\t train=[%.4f (%.4f %.4f)], test=[(%.4f %.4f)]  [%.1f s]" % (epoch + 1, t2 - t1, t2_loss, train_result[0], train_result[1], test_result[0], test_result[1], time() - t2))
                    # f1.write("Epoch %d: [%.1f s]\t train=[%.4f (%.4f %.4f)], test=[(%.4f %.4f)]  [%.1f s]\n" % (epoch + 1, t2 - t1, t2_loss, train_result[0], train_result[1], test_result[0], test_result[1], time() - t2))
                    print("Epoch %d: [%.1f s]\t train-loss=[%.4f, %.4f], test-accuracy=[(%.4f %.4f)]  [%.1f s]" % (epoch + 1, t2 - t1, t1_loss, t2_loss, test_result[0], test_result[1], time() - t2))
                    f1.write("Epoch %d: [%.1f s]\t train-loss=[%.4f, %.4f], test-accuracy=[(%.4f %.4f)]  [%.1f s]\n" % (epoch + 1, t2 - t1, t1_loss, t2_loss, test_result[0], test_result[1], time() - t2))
  
                f1.flush()    
            f1.close()

    def evaluate(self, data):  # evaluate the results for an input set and disable dropout
        # Task 1
        t1_num_example = len(data['R'])
        t1_feed_dict = {self.users: [[u] for u in data['U']], self.items: [[i] for i in data['I']], self.neighbors: [i for i in data['N']], self.ratings: [[R] for R in data['R']]}
        t1_predictions = self.sess.run(self.predictions, feed_dict=t1_feed_dict)
        t1_y_pred = np.reshape(t1_predictions, (t1_num_example,))
        t1_y_true = np.reshape(data['R'], (t1_num_example,))
        t1_predictions_bounded = np.maximum(t1_y_pred, np.ones(t1_num_example) * min(t1_y_true))  # bound the lower values
        t1_predictions_bounded = np.minimum(t1_predictions_bounded, np.ones(t1_num_example) * max(t1_y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(t1_y_true, t1_predictions_bounded))
        MAE = mean_absolute_error(t1_y_true, t1_predictions_bounded)
        return RMSE, MAE


if __name__ == '__main__':
    #===========================================================================
    # for emb in [32]:
    #     for reg_param in [0.001]:
    #         dl = DataLoader('datasets/social/douban/fold1')                    
    #         model = ACF_SE(num_users=dl.num_users, num_items=dl.num_items, num_neighbors=num_neighbors, n_neg_samples=10, reg_param=reg_param, embedding_size=emb,
    #                        layers=[32, 16, 8], epoch=200, t1_batch_size=64, t2_batch_size=128, learning_rate=1e-3, keep_prob=[1, 1, 1],
    #                        verbose=1, optimizer_type='GradientDescentOptimizer')
    #         # model.pre_train_embedding(dl.Link_data, 100)
    #         model.train(dl.Train_data, dl.Test_data, dl.Link_data)
    #===========================================================================
    num_neighbors = 10
    
    for emb in [20]:
        for reg_param in [0.001]:
            dl = DataLoader('../datasets/filmtrust/fold1') 
            dl.extendAllDatawithNeighborhood(num_neighbors)                   
            model = ACF_SE(num_users=dl.num_users, num_items=dl.num_items, num_neighbors=num_neighbors, n_neg_samples=5, reg_param=reg_param, embedding_size=emb,
                           epoch=500, t1_batch_size=256, t2_batch_size=64, learning_rate=1e-3, verbose=1, optimizer_type='GradientDescentOptimizer')
            #model.pre_train_embedding(dl.Link_data, 300)
            model.train(dl.Train_data, dl.Test_data, dl.Link_data)
            
    for emb in [20]:
        for reg_param in [0.005]:
            dl = DataLoader('../datasets/ciaodvd/fold1') 
            dl.extendAllDatawithNeighborhood(10)                   
            model = ACF_SE(num_users=dl.num_users, num_items=dl.num_items, num_neighbors=num_neighbors, n_neg_samples=10, reg_param=reg_param, embedding_size=emb,
                            epoch=400, t1_batch_size=512, t2_batch_size=256, learning_rate=1e-4,verbose=1, optimizer_type='AdamOptimizer')
            #model.pre_train_embedding(dl.Link_data, 100)
            model.train(dl.Train_data, dl.Test_data, dl.Link_data)
            
    for emb in [20]:
        for reg_param in [0.001]:
            dl = DataLoader('../datasets/epinion/fold1')
            dl.extendAllDatawithNeighborhood(10)                  
            model = ACF_SE(num_users=dl.num_users, num_items=dl.num_items, num_neighbors=num_neighbors, n_neg_samples=10, reg_param=reg_param, embedding_size=emb,
                           epoch=200, t1_batch_size=512, t2_batch_size=256, learning_rate=1e-3, verbose=1, optimizer_type='GradientDescentOptimizer')
            # model.pre_train_embedding(dl.Link_data, 200)
            model.train(dl.Train_data, dl.Test_data, dl.Link_data)        
    
