import sys 
sys.path.append("..") 
import glob
import os
import numpy as np
from cofactorization import cofacto
from cofactorization import rec_eval
from  datatool import DataProcessor
os.environ['OPENBLAS_NUM_THREADS'] = '1'

'''
Liang D, Altosaar J, Charlin L, et al. Factorization meets the item embedding: Regularizing matrix factorization with item co-occurrence[C]
//Proceedings of the 10th ACM conference on recommender systems. ACM, 2016: 59-66.
'''
if __name__ == '__main__': 
    
    scale = 0.03
    k_ns = 1
    n_components = 100
    max_iter = 20
    n_jobs = 4
    lam_theta = lam_beta = 1e-5 * scale
    lam_gamma = 1e-5
    c0 = 1. * scale
    c1 = 10. * scale
    
    log = open('log/cofactor.log', 'a') 
    for DATASET_NAME in ['lastfm', 'ml-1m','citeulike','andriod_app','ml-100k']:
    #for DATASET_NAME in ['ml-100k','citeulike', 'ml-100k', 'ml-1m', 'lastfm']:
        log.write("################### ")
        log.write(DATASET_NAME)
        log.write(" ###################\n")
        DATA_DIR = 'datasets/' + DATASET_NAME + '/'
        dp = DataProcessor(DATA_DIR + 'ratings.dat')
        train, test = dp.split_ratings_by_leaveoneout()
        M = dp.est_sppmi_of_items(train)
        train_data = train.tocsr()
        test_data = test.tocsr()
        print(test_data)
        
        for scale in [0.01, 0.1, 1.0, 10.0]:
                log.write("scale=%f  \n" % (scale))
                          
                lam_theta = lam_beta = lam_gamma * scale
                c0 = 1. * scale
                c1 = 10. * scale
                
                save_dir = os.path.join(DATA_DIR, 'ns%d_scale%1.2E' % (k_ns, scale))
                coder = cofacto.CoFacto(n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs,
                                        random_state=2019, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True,
                                        lam_theta=lam_theta, lam_beta=lam_beta, lam_gamma=lam_gamma, c0=c0, c1=c1)
                coder.fit(train_data, M, vad_data=test_data, batch_users=1024, k=10)
                
                n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))
                
                params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                U, V = params['U'], params['V']
                HR_at_1 = rec_eval.recall_at_k(train_data, test_data, U, V, k=1)
                HR_at_5 = rec_eval.recall_at_k(train_data, test_data, U, V, k=5)
                NDCG_at_5 = rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=5)
                NDCG_at_10 = rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=10)
                print ('Test Recall@1: %.4f' % HR_at_1)
                print ('Test Recall@5: %.4f' % HR_at_5)
                print ('Test NDCG@5: %.4f' % NDCG_at_5)
                print ('Test NDCG@10: %.4f' % NDCG_at_10)
                log.write ('Test Recall@1: %.4f \n' % HR_at_1)
                log.write  ('Test Recall@5: %.4f \n' % HR_at_5)
                log.write  ('Test NDCG@5: %.4f \n' % NDCG_at_5)
                log.write  ('Test NDCG@10: %.4f \n' % NDCG_at_10)
                log.flush()
    log.close()
