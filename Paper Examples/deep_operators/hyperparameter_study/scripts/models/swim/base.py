import abc
import numpy as np


def numpy_relative_l2_error(predictions, y):
    predictions = predictions.reshape(predictions.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    diff_norm = np.linalg.norm(predictions - y, axis=-1)
    y_norm = np.linalg.norm(y, axis=-1)
    return diff_norm / y_norm


class BaseSwim(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_params, seed):
        self.predictor = None

    def train(self, train_data, *_):
        train_u0, train_u1 = train_data
        self.predictor.fit(train_u0, train_u1)
        # No iterative metrics
        return {}, []

    def evaluate(self, data):
        u0, u1 = data
        predictions = self.predictor.transform(u0)
        rel_l2_error = numpy_relative_l2_error(predictions, u1)
        return predictions, np.mean(rel_l2_error)