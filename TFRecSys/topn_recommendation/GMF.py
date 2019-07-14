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
from sampler import RatingSampler, PairSampler
from datatool import DataProcessor

                        
class GMF:
    '''
    He X, Liao L, Zhang H, et al. Neural collaborative filtering, WWW2017: 173-182.
    '''

    def __init__(self, n_users, n_items, n_negatives, embed_dim=50, learning_rate=0.001, reg=[0, 0], random_seed=2018):
        self.n_users = n_users
        self.n_items = n_items
        self.n_negatives=n_negatives
        self.embed_dim = embed_dim
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
            self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
            self.negative_samples = tf.placeholder(tf.int32, [None, None])
            self.score_user_ids = tf.placeholder(tf.int32, [None])
            self.score_n_users = tf.placeholder(tf.int32, shape=())
            self.maxk = tf.placeholder(tf.int32, shape=())
            # Variables.
            self.weights = { 
                'user_embeddings':tf.Variable(tf.random_normal([self.n_users, self.embed_dim], mean=0.0, stddev=0.01, dtype=tf.float32)),
                'item_embeddings':tf.Variable(tf.random_normal([self.n_items, self.embed_dim], mean=0.0, stddev=0.01, dtype=tf.float32))
                }
            #define Prediction layer
            self.predict = tf.layers.Dense(units=1, activation='sigmoid', kernel_initializer='lecun_uniform')
            #perform computation
            user_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_positive_items_pairs[:, 0])
            pos_item_emb = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.user_positive_items_pairs[:, 1])
            #for positive user-item pair
            pos_predictions = self.predict.apply(tf.multiply(user_emb, pos_item_emb))
            self.loss = -tf.reduce_sum(tf.log(pos_predictions))* self.n_negatives
            #for negative user-item pairs
            neg_item_embs = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.negative_samples)
            neg_predictions = self.predict.apply(tf.multiply(tf.expand_dims(user_emb, 1), neg_item_embs))
            self.loss += -tf.reduce_sum(tf.log(1 - neg_predictions))
            #optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
            '''
            for evaluation part
            '''
            test_user_emb = tf.expand_dims(tf.nn.embedding_lookup(self.weights['user_embeddings'], self.score_user_ids), 1)
            test_item_emb = tf.expand_dims(self.weights['item_embeddings'], 0)
            
            self.item_scores = tf.squeeze(self.predict.apply(tf.multiply(test_user_emb, test_item_emb)))
            
            self.top_k = tf.nn.top_k(self.item_scores, self.maxk)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()

    def train(self, log, sampler, train, test, iters, top_k):
        
        log.write("embed_dim=%d \n" % (self.embed_dim))
        log.write("############################################### \n")

        test_users = list(set(test.nonzero()[0]))
        eval = Evaluator(train, test)
        for i in range(iters):
            print("Iter %d..." % (i + 1))
            log.write("Iter %d...\t" % (i + 1))

            t1 = time()             
            print("Optimizing  loss...") 
            loss = 0
            sampler.generate_batches()
            while not sampler.is_empty():
                ui_pos, ui_neg, _ = sampler.next_batch()
                _, loss = self.sess.run((self.optimizer, self.loss), {self.user_positive_items_pairs: ui_pos,
                                                                      self.negative_samples: ui_neg})
                loss += loss
            t2 = time()
            
            print("\nTraining loss: %f  in %.2fs" % (loss, t2 - t1))
            log.write("Training loss: %f  in %.2fs \n" % (loss, t2 - t1))
            
            if (i + 1) % 1 == 0:
                test_recall = list()
                test_precision = list()
                test_ndcg = list()
                test_hitratio = list()
                # compute metrics in chunks to utilize speedup provided by Tensorflow
                for user_chunk in toolz.partition_all(50, test_users):
                    _r, _p, _n, _h = eval.evalRankPerformance(self, user_chunk, top_k)
                    test_recall.extend(_r)
                    test_precision.extend(_p)
                    test_ndcg.extend(_n)
                    test_hitratio.extend(_h)

                for j in range(len(top_k)):
                    recall = 0
                    for m in range(len(test_recall)):
                        recall += test_recall[m][j]
                    precision = 0
                    for m in range(len(test_precision)):
                        precision += test_precision[m][j]
                    ndcg = 0
                    for m in range(len(test_ndcg)):
                        ndcg += test_ndcg[m][j]
                    hitratio = 0
                    for m in range(len(test_hitratio)):
                        hitratio += test_hitratio[m][j]
                    print("Top-%d Recall: %f Precision: %f NDCG: %f HR:%f\n" % (top_k[j], recall / len(test_recall), precision / len(test_precision), ndcg / len(test_ndcg), hitratio / len(test_hitratio)))
                    log.write("Top-%d Recall: %f Precision: %f NDCG: %f HR:%f\n" % (top_k[j], recall / len(test_recall), precision / len(test_precision), ndcg / len(test_ndcg), hitratio / len(test_hitratio)))
                t3 = time()
                print("Eval costs: %f s\n" % (t3 - t2))
                log.write("Eval costs: %f s\n" % (t3 - t2))

            log.flush()


if __name__ == '__main__':
    BATCH_SIZE = [64, 64, 64, 64, 64]
    TOP_K = [1, 5, 10]
    MAX_ITERS = [30, 20, 20, 100, 100]
    N_NEGATIVE = [4, 4, 4, 4, 4]
    DATASET_NAMES = ['ml-100k','lastfm', 'ml-1m', 'kindle-store', 'epinion']
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.txt')
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_leaveoneout()
        sampler = RatingSampler(train, batch_size=BATCH_SIZE[i], n_negative=N_NEGATIVE[i], check_negative=True)   
        
        log = open('../log/' + DATASET_NAMES[i] + '.GMF.log', 'a') 
        log.write("############################################### \n")
        log.write("n_negative=%d  \n" % (N_NEGATIVE[i]))
        log.write("batch_size=%d \n" % (BATCH_SIZE[i]))
        log.flush()      
        for embed_dim in [8, 16, 32, 64]:
            model = GMF(n_users=n_users, n_items=n_items, n_negatives=N_NEGATIVE[i], embed_dim=embed_dim, learning_rate=0.001)
            model.train(log, sampler, train, test, MAX_ITERS[i], TOP_K)
        log.close()
