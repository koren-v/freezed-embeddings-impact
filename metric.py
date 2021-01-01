import numpy as np

from torch import nn
from sklearn.metrics import f1_score


class F1Metric(nn.Module):
    def __init__(self, average="macro"):
        super(F1Metric, self).__init__()
        self.average = average

    def forward(self, y_true, y_pred):
        pred_labels = np.argmax(y_pred, axis=-1)
        return f1_score(y_true=y_true,
                        y_pred=pred_labels,
                        average=self.average,
                        zero_division=0.0)
