import sys 
sys.path.append("..") 
import functools
import numpy
import tensorflow as tf
import numpy as np
from time import time
import toolz
from tqdm import tqdm
from evaluator import Evaluator
from sampler import RatingSampler, TrustSampler
from datatool import socialdata, split_data_by_leave_one_out, est_social_similarity, est_rating_similarity
from scipy.sparse import dok_matrix, lil_matrix


class SRCML:
    '''
    Zhengxin Zhang, Qimin Zhou, Hua Zhao, Hao Wu, Social Regularized Collaborative Metric Learning, DSS2019.
    '''
    def __init__(self, n_users, n_items, embed_dim=50, margin=1.5, layers=[100, 100], master_learning_rate=0.001,
                 clip_norm=1.0, use_rank_weight=True, use_cov_loss=False, cov_loss_weight=10, social_lambda=1.0, random_seed=2018):
        """
        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature trust_loss(default: None)
        :param margin: hinge trust_loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        """

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.social_lambda = social_lambda
        self.clip_norm = clip_norm
        self.margin = margin
        self.layers = layers
        
        self.master_learning_rate = master_learning_rate
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, rating_loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
            self.negative_samples = tf.placeholder(tf.int32, [None, None])
            self.score_user_ids = tf.placeholder(tf.int32, [None])
        
            self.trust_pairs = tf.placeholder(tf.int32, [None, 2])
            #self.positions = tf.placeholder(tf.int32, [None])
            self.sims=tf.placeholder(tf.float32, [None, 1])
            # Variables.
            self.weights = self._initialize_weights()
            
            '''
            Subgraph 1
            '''
            # user embedding (N, K)
            users = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_positive_items_pairs[:, 0], name="users")
            # positive item embedding (N, K)
            pos_items = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.user_positive_items_pairs[:, 1], name="pos_items")
            
            # positive item to user distance (N)
            pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")
    
            # negative item embedding (N, K, W)
            neg_items = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.negative_samples, name="neg_items")
            # neg_items=self.item_projection(neg_items)
            neg_items = tf.transpose(neg_items, (0, 2, 1))
            # distance to negative items (N x W)
            distance_to_neg_items = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1, name="distance_to_neg_items")
    
            # best negative item (among W negative samples) their distance to the user embedding (N)
            closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")
    
            # compute hinge rating_loss (N)
            loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0, name="pair_loss")
    
            if self.use_rank_weight:
                # indicator matrix for impostors (N x W)
                impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
                # approximate the rank of positive item by (number of impostor / W per user-positive pair)
                rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
                # apply rank weight
                loss_per_pair *= tf.log(rank + 1)
    
            # the embedding trust_loss
            self.rating_loss = tf.reduce_sum(loss_per_pair, name="rating_loss")

            self.loss = self.rating_loss
            if self.use_cov_loss:
                self.loss += self.covariance_loss()
                
            '''
            #########################
            # social regularization #
            #########################
            
            subloss=\sum_{i,j} social_lambda*sim(i,j)||u_i-u_j||^2 
            social_lambda: regularization paramter, when social_lambda=0,  no social network information is considered.
            sim(i,j): similarity value between user i and user j
            ||u_i-u_j||^2: Eculid distance between user i and user j
            
            For fast computation, we only take into account those users i or j who are in the current mini_batch, by using tf.unique and tf.gather operations!
            '''
            #trust_pairs=tf.gather(self.trust_pairs, self.positions)
            #sims =tf.gather(self.sims, self.positions)
            
            
            emb_trustors = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.trust_pairs[:,0], name="trustors")
            emb_trustees = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.trust_pairs[:,1], name="trustees")
            
            trust_distances = tf.squared_difference(emb_trustors, emb_trustees)
    
            self.trust_loss = tf.reduce_sum(tf.multiply(self.sims, trust_distances))
            
            #social regularization
            self.loss += self.social_lambda * self.trust_loss
            
            self.optimizer = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.loss)
        
            with tf.control_dependencies([self.optimizer]):
                tf.assign(self.weights['user_embeddings'], tf.clip_by_norm(self.weights['user_embeddings'], self.clip_norm, axes=[1]))
                tf.assign(self.weights['item_embeddings'], tf.clip_by_norm(self.weights['item_embeddings'], self.clip_norm, axes=[1]))
                
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()

    def _initialize_weights(self):
        all_weights = dict()
        # _________ Embedding Layer _____________
        all_weights['user_embeddings'] = tf.Variable(tf.random_normal([self.n_users, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        all_weights['item_embeddings'] = tf.Variable(tf.random_normal([self.n_items, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
       
        # ________ Perception Layers __________
        #=======================================================================
        # if len(self.layers) > 0:
        #     for i in range(1, len(self.layers)):
        #         glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
        #         all_weights['layer_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
        #         all_weights['bias_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
        # 
        #=======================================================================
        return all_weights
   
    def covariance_loss(self):
        X = tf.concat((self.weights['item_embeddings'], self.weights['user_embeddings']), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight
    
    def item_scores(self):
        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(self.weights['user_embeddings'], self.score_user_ids), 1)
        # (1, N_ITEM, K)
        item = tf.expand_dims(self.weights['item_embeddings'], 0)
        # score = minus distance (N_USER, N_ITEM)
        return -tf.reduce_sum(tf.squared_difference(user, item), 2)

    def train(self, log, dataset_name, sampler1, sampler2, train, test, iters, top_k):
        """
        Optimize the self. TODO: implement early-stopping
        :param dataset_name: dataset used
        :param sampler: mini-batch sampler for rating data
        :param train: train user-item matrix
        :param test: test user-item matrix
        :param iters: max number of iterations
        :param top_k: top-k in performance metrics
        :return: None
        """
        log.write("embed_dim=%d \n" % (self.embed_dim))
        log.write("margin=%.2f \n" % (self.margin))
        log.write("social_lambda=%.2f \n" % (self.social_lambda))
        log.write("############################################### \n")

        # sample some users to calculate recall validation
        # test_users = numpy.random.choice(list(set(test.nonzero()[0])), size=1000, replace=True)
        test_users = list(set(test.nonzero()[0]))
        eval = Evaluator(train, test)
        
        for i in range(iters):
            print("Iter %d..." % (i + 1))
            log.write("Iter %d...\t" % (i + 1))
            
            # TODO: early stopping based on validation recall
            # train self.
            losses = []
            # run n mini-batches
            t1 = time()
            print("Optimizing  rating_loss...") 
            sampler1.generate_bacthes()
            while not sampler1.is_empty():
                ui_pos, ui_neg = sampler1.next_batch()
                trust_pairs, pair_sims =  sampler2.next_batch(ui_pos[:,0])
                _, loss = self.sess.run((self.optimizer, self.rating_loss), {self.user_positive_items_pairs: ui_pos,
                                                                              self.negative_samples: ui_neg,
                                                                              #self.positions:u_positions,
                                                                              self.trust_pairs:trust_pairs,
                                                                              self.sims:pair_sims})
                losses.append(loss)
            
            t2 = time()
            print("\nTraining loss: %f in %.2fs" % (numpy.sum(losses), t2 - t1))
            log.write("Training loss: %f in %.2fs\n" % (numpy.sum(losses), t2 - t1))
            
            if (i + 1) % 20 == 0:
                
                for k in top_k:
                    test_recall = []
                    test_precision = []
                    test_ndcg = []
                    # compute metrics in chunks to utilize speedup provided by Tensorflow
                    for user_chunk in toolz.partition_all(100, test_users):
                        r, p, n = eval.evalRankPerformance(self, user_chunk, k)
                        test_recall.extend(r)
                        test_precision.extend(p)
                        test_ndcg.extend(n)
                    recall = numpy.mean(test_recall)
                    precision = numpy.mean(test_precision)
                    ndcg = numpy.mean(test_ndcg)
                    print("Top-%d Recall: %f Precision: %f NDCG: %f\n" % (k, recall, precision, ndcg))
                    log.write("Top-%d Recall: %f Precision: %f NDCG: %f\n" % (k, recall, precision, ndcg))
            
            log.flush()


if __name__ == '__main__':
    BATCH_SIZE = [1024, 1024, 1024, 1024, 1024]
    TOP_K = [5, 10, 20, 50]
    MAX_ITERS = [100,200,50,200,150]
    N_NEGATIVE = [20, 20, 20, 20, 20]
    DATASET_NAMES = ['filmtrust','lastfm', 'douban', 'ciaodvd', 'epinion']
    #
    #split_ratio=(8, 2)
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        user_item_matrix, trust_matrix = socialdata(DATASET_NAMES[i])
        n_users, n_items = user_item_matrix.shape
        # get train/valid/test user-item matrices
        train, test = split_data_by_leave_one_out(user_item_matrix)
        social_sim = est_social_similarity(trust_matrix)
        rating_sim = est_rating_similarity(train, social_sim)
        # create warp samplers
        sampler1 = RatingSampler(train, batch_size=BATCH_SIZE[i], n_negative=N_NEGATIVE[i], check_negative=True)
        sampler2 =TrustSampler(rating_sim) 
        log = open('log/' + DATASET_NAMES[i] + '.srcml.log', 'a')
        log.write("############################################### \n")
        #log.write("split_ratio=(%d, %d)  \n" %(split_ratio[0],split_ratio[1]))
        log.write("n_negative=%d \n" % (N_NEGATIVE[i]))
        log.write("batch_size=%.2f \n" % (BATCH_SIZE[i]))      
        log.flush()  
#         for embed_dim in [50, 100, 200]:
#             for margin in [0.5, 1.0, 1.5]:
        for embed_dim in [100]:
            for margin in [2.0]:
                for social_lambda in [0, 0.5]:
                        # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
                        # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
                        model = SRCML(n_users, n_items, embed_dim=embed_dim, layers=[embed_dim, int(embed_dim / 2), int(embed_dim / 2)], margin=margin, social_lambda=social_lambda)
                        model.train(log, DATASET_NAMES[i], sampler1,  sampler2, train, test,  MAX_ITERS[i], TOP_K)
        log.close()