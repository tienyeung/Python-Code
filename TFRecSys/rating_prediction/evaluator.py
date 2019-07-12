import tensorflow as tf
import numpy as np
from sampler import PairSampler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Evaluator(object):

    def __init__(self, test_user_item_matrix, batch_size=1024):
        """
        Create a evaluator for the task of rating prediction
        :param test_user_item_matrix: the held-out user-item pairs we make prediction
        """
        self.test_user_item_matrix = test_user_item_matrix
        self.sampler = PairSampler(self.test_user_item_matrix, batch_size)
        
    def evalRatingPerformance(self, model, dropout_keep=None):
        """
        Compute the performance metrics of rating predictions
        :param model: the model we are going to evaluate
        :param dropout_keep: an array to disable/enable dropout in deep neural networks
        :return: MAE, MSE, RMSE
        """
        self.sampler.generate_batches(shuffle=False)
        
        all_predictions = list()
        all_ratings = list()
        
        while not self.sampler.is_empty():
            users, items, ratings = self.sampler.next_batch()
            if dropout_keep is None:
                predictions = model.sess.run(model.out, {model.users: users, model.items: items})
            else:
                predictions = model.sess.run(model.out, {model.users: users, model.items: items, model.dropout_keep:dropout_keep})
            num_example = len(ratings)
            predictions_bounded = np.maximum(np.squeeze(predictions), np.ones(num_example) * min(ratings))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(ratings))  # bound the higher values
            all_ratings.extend(ratings)
            all_predictions.extend(predictions_bounded)
        
        error = np.subtract(all_predictions, all_ratings)
        mae = np.mean(np.abs(error))
        mse = np.mean(np.square(error))
        return mae, mse, np.sqrt(mse)
