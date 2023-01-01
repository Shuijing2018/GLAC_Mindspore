from mindspore import nn
from mindspore.ops import functional as F
import mindspore

def gce_loss(outputs, Y, q):
    sm_outputs = F.log_softmax(outputs, axis=1)
    pow_outputs = F.pow(-sm_outputs, q)
    sample_loss = mindspore.numpy.sum((1 - (pow_outputs * Y)), axis=1) / q
    return sample_loss