import tensorflow as tf
from scipy.sparse import lil_matrix
import numpy


class Evaluator(object):

    def __init__(self, train_user_item_matrix, test_user_item_matrix, max_k=500):
        """
        Create a evaluator for recall@K evaluation
        :param model: the model we are going to evaluate
        :param train_user_item_matrix: the user-item pairs used in the training set. These pairs will be ignored
               in the recall calculation
        :param test_user_item_matrix: the held-out user-item pairs we make prediction against
        """
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}
            self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
        else:
            self.max_train_count = 0
        
        self.idcg = numpy.zeros(max_k+1)
        
        for index in range(max_k) :
            self.idcg[index + 1] = (1.0 / numpy.log2(index + 2)) + self.idcg[index]
                      
    def evalRankPerformance(self, model, users, k=[10]):
        """
        Compute the Top-K precision, recall, ndcg for a particular user given the predicted scores to items
        :param users: the users to eval the recall
        :param k: compute the precision, recall, ndcg for the top K items
        :return: P@K,R@K,nDCG@K
        """
        # compute the top (K +  Max Number Of Training Items for any user) items for each user
        _, user_tops = model.sess.run(model.top_k,
                                {model.score_user_ids: users, 
                                 model.score_n_users: len(users),
                                 model.maxk:k[-1] + self.max_train_count})
        recall,precision,ndcg,hitratio = list(),list(),list(),list()
        
        for user_id, tops in zip(users, user_tops):
            train_set = self.user_to_train_set.get(user_id, set())
            test_set = self.user_to_test_set.get(user_id, set())
            top_n_items = 0
            hits = 0
            dcg = 0
            r,p,n,h=list(),list(),list(),list()
            for i in tops:
                # ignore item in the training set
                if i in train_set: continue
                elif i in test_set:
                    hits += 1
                    dcg += 1.0 / numpy.log2(top_n_items + 2) 
                top_n_items += 1
                for _k in k:
                    if top_n_items == _k:
                        if len(test_set) < _k: n.append(dcg / self.idcg[len(test_set)])
                        else: n.append(dcg / self.idcg[_k])
                
                        r.append(hits / float(len(test_set)))
                        p.append(hits / float(_k))
                        h.append(hits)
                if top_n_items == k[-1]:  break
                
            recall.append(r)
            precision.append(p)
            ndcg.append(n)
            hitratio.append(h)
            
        return recall, precision, ndcg, hitratio
