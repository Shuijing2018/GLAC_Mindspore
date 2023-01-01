import mindspore.nn as nn
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import mindspore
from mindspore.ops import operations as ops
zeros = ops.Zeros()
from mindspore import Model
from mindspore.train.callback import Callback, LossMonitor

class CustomWithLossCell(nn.Cell):
    """Connect the forward network and the loss function"""

    def __init__(self, backbone, loss_fn, theta, args):
        """There are two inputs, the forward network backbone and the loss function"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._theta = theta
        self._args = args

    def construct(self, batchX, batchY, index):
        label_sum_for_each_instance = mindspore.numpy.sum(batchY, axis=1)  # number of examples in the batch
        unlabeled_index = []
        labeled_index = []
        for i in range(len(label_sum_for_each_instance)):
            if label_sum_for_each_instance[i] == 0:
                unlabeled_index.append(i)
            else:
                labeled_index.append(i)
        unlabeled_index = mindspore.Tensor(unlabeled_index)
        labeled_index = mindspore.Tensor(labeled_index)

        pred = self._backbone(batchX.astype(np.float32))

        pred_labeled = pred[labeled_index, :]
        labeled_batchY = batchY[labeled_index, :]
        pred_unlabeled = pred[unlabeled_index, :]
        unlabeled_batchY = batchY[unlabeled_index, :]
        unlabeledY = mindspore.numpy.zeros_like(unlabeled_batchY)
        unlabeledY[:, -1] = 1

        n, k = labeled_batchY.shape[0], labeled_batchY.shape[1]
        temp_loss = []
        for i in range(k):
            tempY = zeros((n, k), mindspore.float32)
            tempY[:, i] = 1.0  # calculate the loss with respect to the i-th label
            temp_loss.append(self._loss_fn(pred_labeled, tempY, self._args.q))
        temp_loss = mindspore.numpy.stack(temp_loss, axis=1)
        loss_1_pos = mindspore.numpy.mean(self._theta * mindspore.numpy.sum(temp_loss * labeled_batchY, axis=1))
        loss_1_neg = mindspore.numpy.mean(self._theta * temp_loss[:, -1])

        loss_2 = mindspore.numpy.mean(self._loss_fn(pred_unlabeled, unlabeledY, self._args.q))

        ##

        if loss_2 - loss_1_neg >= 0:
            train_loss = loss_1_pos + loss_2 - loss_1_neg
        else:
            train_loss = loss_1_pos + loss_2 - loss_1_neg + self._args.lamda * (loss_1_neg - loss_2) ** self._args.t

        return train_loss
        # return 0


def NRPR(train_dataloader, X_test, y_test, model, optimizer, loss_fn, theta, args):
    custom_model = CustomWithLossCell(model, loss_fn, theta, args)
    model = Model(custom_model, optimizer=optimizer)
    model.train(args.ep, train_dataloader, callbacks=[LossMonitor(1), ])

