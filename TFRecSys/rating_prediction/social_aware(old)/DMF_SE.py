import argparse, math
from time import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error

from DataTool import DataLoader
import numpy as np
import tensorflow as tf


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DMF_SE.")
    parser.add_argument('--layers[0]', type=int, default=64, help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[64,64]', help="Size of each perception layer.")
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.8]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout. ')
    parser.add_argument('--optimizer', nargs='?', default='GradientDescentOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')

    #    parser.add_argument('--early_stop', type=int, default=1, help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()


class DMF_SE():

    def __init__(self, num_users, num_items, n_neg_samples, reg_param, layers, epoch, t1_batch_size, t2_batch_size, learning_rate, keep_prob, optimizer_type, verbose, random_seed=2018):
        # bind params to class
        self.t1_batch_size = t1_batch_size
        self.t2_batch_size = t2_batch_size
        # self.t2_layers[0] = t2_layers[0]
        self.layers = layers  # perception_layers
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.num_users = num_users
        self.num_items = num_items
        self.n_neg_samples = n_neg_samples
        self.reg_param = np.array(reg_param)
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
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.users = tf.placeholder(tf.int32, shape=[None, 1])  # None * 1
            self.items = tf.placeholder(tf.int32, shape=[None, 1])  # None * 1
            self.ratings = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            
            # Variables.
            self.weights = self._initialize_weights()

            ###########################################
            #     Model for rating prediction         #    
            ###########################################
            
           # _________ Embedding Layer _____________
            self.u_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.users)  # None * 1* layers[0]
            self.i_emb = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.items)  # None * 1* layers[0]

            self._u_emb = tf.reshape(self.u_emb, shape=[-1, self.layers[0]])
            self._i_emb = tf.reshape(self.i_emb, shape=[-1, self.layers[0]])
            
            # ________ Perception Layers __________
            for i in range(1, len(self.layers)):
                self._u_emb = tf.add(tf.matmul(self._u_emb, self.weights['layer_user_%d' % i]), self.weights['bias_user_%d' % i])  # None * layer[i] * 1
                self._u_emb = tf.nn.dropout(tf.nn.relu(self._u_emb), self.dropout_keep[i])  # dropout at each Deep layer
                
                self._i_emb = tf.add(tf.matmul(self._i_emb, self.weights['layer_item_%d' % i]), self.weights['bias_item_%d' % i])  # None * layer[i] * 1
                self._i_emb = tf.nn.dropout(tf.nn.relu(self._i_emb), self.dropout_keep[i])  # dropout at each Deep layer

            # ________ Prediction Layer __________    
            self.predictions = tf.reduce_sum(tf.multiply(self._u_emb, self._i_emb), 1, keep_dims=True)
            
            # Compute the loss.
            self.t1_loss = tf.nn.l2_loss(tf.subtract(self.ratings, self.predictions))
            if self.reg_param[0] > 0:
                regularizer = tf.contrib.layers.l2_regularizer(self.reg_param[0])(self.i_emb)
                
                if self.reg_param[1] > 0:
                    self._node_emb = tf.nn.embedding_lookup(self.weights['node_embeddings'], self.users) 
                    regularizer = regularizer + tf.contrib.layers.l2_regularizer(self.reg_param[1])(tf.subtract(self.u_emb, self._node_emb))
                
                for i in range(1, len(self.layers)):
                     regularizer = regularizer + tf.contrib.layers.l2_regularizer(self.reg_param[0])(self.weights['layer_user_%d' % i]) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param[0])(self.weights['bias_user_%d' % i]) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param[0])(self.weights['layer_item_%d' % i]) + \
                                tf.contrib.layers.l2_regularizer(self.reg_param[0])(self.weights['bias_item_%d' % i])
                self.t1_loss = self.t1_loss + regularizer

            ###########################################
            #     Model for social network embedding  #    
            ###########################################
            
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.nodes = tf.placeholder(tf.int32, shape=[None])  # None * 1
            
            self.node_emb = tf.nn.embedding_lookup(self.weights['node_embeddings'], self.nodes)  # batch_size * layers[0]

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
        
        all_weights['user_embeddings'] = tf.Variable(tf.random_uniform([self.num_users, self.layers[0]], -1.0, 1.0), name='user_embeddings')  # features_U* K
        all_weights['item_embeddings'] = tf.Variable(tf.random_uniform([self.num_items, self.layers[0]], -1.0, 1.0), name='item_embeddings')  # features_I * K
        all_weights['node_embeddings'] = tf.Variable(tf.random_uniform([self.num_users, self.layers[0]], -1.0, 1.0), name='node_embeddings')  # features_U* K
        
        if len(self.layers) > 0:
            for i in range(1, len(self.layers)):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_user_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_user_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]

            for i in range(1, len(self.layers)):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_item_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_item_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
    
        all_weights['out_embeddings'] = tf.Variable(tf.random_uniform([self.num_users, self.layers[0]], -1.0, 1.0))
        all_weights['biases'] = tf.Variable(tf.zeros([self.num_users]))
        
        return all_weights

    def t1_partial_fit(self, data):  # fit a batch and enable dropout
        feed_dict = {self.users: data['U'], self.items: data['I'], self.ratings: data['R'], self.dropout_keep: self.keep_prob}
        loss, opt = self.sess.run((self.t1_loss, self.t1_optimizer), feed_dict=feed_dict)
        return loss

    def t2_partial_fit(self, data):  # fit a batch
        feed_dict = {self.nodes: data['U'], self.labels: data['V']}
        loss, opt = self.sess.run((self.t2_loss, self.t2_optimizer), feed_dict=feed_dict)
        return loss

    @staticmethod
    def get_random_block_from_t1_data(data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['R']) - batch_size)
        U, I, R = [], [], []
        # forward get sample
        i = start_index
        while len(U) < batch_size and i < len(data['U']):
            R.append([data['R'][i]])
            U.append([data['U'][i]])
            I.append([data['I'][i]])
            i = i + 1
        # backward get sample
        i = start_index
        while len(U) < batch_size and i >= 0:
            R.append([data['R'][i]])
            U.append([data['U'][i]])
            I.append([data['I'][i]])
            i = i - 1
        return {'U': U, 'I': I, 'R': R}

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
    def shuffle_in_unison_scary(a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
    
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
            f1 = open('../log/dmfse.log', 'a')
            f1.write("### DMF_SE ###\n")
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
            
    def train(self, Train_data, Test_data, Link_data, log):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            # Write configurations in log
            f1 = open(log, 'a')
            f1.write("layers: size=%s, dropout=%s \n" % (self.layers, self.keep_prob))
            f1.write("Regularization: %s \n" %(self.reg_param))
            t2 = time()
            # init_train = self.evaluate(Train_data)
            # init_test = self.evaluate(Test_data)
            # print("Initialization: \t train=[(%.4f %.4f)], test=[(%.4f %.4f)]  [%.1f s]" % (init_train[0], init_train[1], init_test[0], init_test[1], time() - t2))
            # f1.write("Initialization: \t train=[(%.4f %.4f)], test=[(%.4f %.4f)]  [%.1f s]\n" % (init_train[0], init_train[1], init_test[0], init_test[1], time() - t2))

            for epoch in range(self.epoch):
                t1 = time()
                t2_loss = 0
                if self.reg_param[1]>0:
                    # learning social network embedding
                    assign_op = tf.assign(self.weights['node_embeddings'], self.weights['user_embeddings'])
                    self.sess.run(assign_op)
                    rng_state = np.random.get_state()
                    np.random.shuffle(Link_data['U'])
                    np.random.set_state(rng_state)
                    np.random.shuffle(Link_data['V'])
                    total_batch = int(len(Link_data['U']) / self.t2_batch_size)
                    t2_loss = 0
                    for i in range(total_batch):
                        batch_xs = self.get_random_block_from_t2_data(Link_data, self.t2_batch_size)
                        t2_loss = t2_loss + self.t2_partial_fit(batch_xs)
                            
                # training deep matrix factorization model
                self.shuffle_in_unison_scary(Train_data['U'], Train_data['I'], Train_data['R'])
                total_batch = int(len(Train_data['R']) / self.t1_batch_size)
                t1_loss = 0
                for i in range(total_batch):
                    batch_xs = self.get_random_block_from_t1_data(Train_data, self.t1_batch_size)
                    t1_loss = t1_loss + self.t1_partial_fit(batch_xs)
                    
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
        t1_feed_dict = {self.users: [[u] for u in data['U']], self.items: [[i] for i in data['I']], self.ratings: [[R] for R in data['R']], self.dropout_keep: self.no_dropout}
        t1_predictions = self.sess.run(self.predictions, feed_dict=t1_feed_dict)
        t1_y_pred = np.reshape(t1_predictions, (t1_num_example,))
        t1_y_true = np.reshape(data['R'], (t1_num_example,))
        t1_predictions_bounded = np.maximum(t1_y_pred, np.ones(t1_num_example) * min(t1_y_true))  # bound the lower values
        t1_predictions_bounded = np.minimum(t1_predictions_bounded, np.ones(t1_num_example) * max(t1_y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(t1_y_true, t1_predictions_bounded))
        MAE = mean_absolute_error(t1_y_true, t1_predictions_bounded)
        return RMSE, MAE


if __name__ == '__main__':
    for reg1 in [0.01, 0.1, 1.0]:
        for reg_social in [0, 1.0, 10.0, 100.0]:
            dl = DataLoader('../datasets/douban/fold1')                    
            model = DMF_SE(num_users=dl.num_users, num_items=dl.num_items, n_neg_samples=5, reg_param=[reg1,reg_social], 
                           layers=[64,32,16], epoch=100, t1_batch_size=256, t2_batch_size=256, learning_rate=1e-3, keep_prob=[1, 1, 1],
                           verbose=1, optimizer_type='AdamOptimizer')
            # model.pre_train_embedding(dl.Link_data, 100)
            model.train(dl.Train_data, dl.Test_data, dl.Link_data, '../log/dmfse.douban.log')
    for reg1 in [0.01, 0.1, 1.0]:
        for reg_social in [0, 1.0, 10.0, 100.0]:
            dl = DataLoader('../datasets/ciaodvd/fold1')                    
            model = DMF_SE(num_users=dl.num_users, num_items=dl.num_items, n_neg_samples=10, reg_param=[reg1, reg_social], 
                           layers=[64, 32, 16], epoch=100, t1_batch_size=256, t2_batch_size=256, learning_rate=1e-4, keep_prob=[1, 1,1],
                           verbose=1, optimizer_type='AdamOptimizer')
            # model.pre_train_embedding(dl.Link_data, 100)
            model.train(dl.Train_data, dl.Test_data, dl.Link_data,'../log/dmfse.ciaodvd.log')
    for reg1 in [0.01, 0.1, 1.0]:
        for reg_social in [0, 1.0, 10.0, 100.0]:
            dl = DataLoader('../datasets/epinion/fold1')                    
            model = DMF_SE(num_users=dl.num_users, num_items=dl.num_items, n_neg_samples=10, reg_param=[reg1, reg_social], 
                           layers=[64, 32, 16], epoch=100, t1_batch_size=256, t2_batch_size=256, learning_rate=1e-4, keep_prob=[1, 1,1],
                           verbose=1, optimizer_type='AdamOptimizer')
            # model.pre_train_embedding(dl.Link_data, 200)
            model.train(dl.Train_data, dl.Test_data, dl.Link_data,'../log/dmfse.ciaodvd.log')        
    
