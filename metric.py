import numpy as np

from torch import nn
from sklearn.metrics import accuracy_score


class AccuracyMetric(nn.Module):
    def forward(self, y_true, y_pred):
        pred_labels = np.argmax(y_pred, axis=-1)
        return accuracy_score(y_true=y_true,
                              y_pred=pred_labels)
