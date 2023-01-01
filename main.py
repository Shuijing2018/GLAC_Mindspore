from utils_data import prepare_cv_datasets, make_cv_mpu_train_set, make_uci_mpu_train_set, gen_index_dataset
from utils_loss import gce_loss
import argparse
from utils_model import linear_model, mlp_model
from algorithms import NRPR
from utils_func import KernelPriorEstimator
import numpy as np
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import Callback, LossMonitor
from mindspore import Tensor, set_context, PYNATIVE_MODE, dtype as mstype

# Set the operation mode to dynamic graph mode
set_context(mode=PYNATIVE_MODE)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=1e-2)
parser.add_argument('-wd', type=float, default=1e-4)
parser.add_argument('-ds', type=str, help='specify a dataset', default='har')
parser.add_argument('-uci', type=int, help='UCI dataset or not', default=1, choices=[1,0])
parser.add_argument('-ep', type=int, default=10)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-mo', type=str, help='specify a model', default='linear', choices=['mlp','linear'])
parser.add_argument('-iter', type=int, default=50)
parser.add_argument('-bs', type=int, default=500)
parser.add_argument('-lamda', type=float, default=1.2)
parser.add_argument('-q', type=float, default=0.1)
parser.add_argument('-t', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.seed)

        
# number of labeled, unlabeled, and test per-class
if args.ds == 'mnist':
    num_labeled = 4000
    num_unlabeled = 1000
    num_test = 100      


if args.uci ==1:
    X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, X_test, y_test = make_uci_mpu_train_set(args.ds, 500, 1000, 1000, args.seed)
else:
    full_train_loader, full_test_loader = prepare_cv_datasets(args.ds)
    X_labeled, Y_labeled, X_unlabeled, Y_unlabeled, X_test, y_test = make_cv_mpu_train_set(full_train_loader, full_test_loader, num_labeled, num_unlabeled, num_test, args)

# create the zero label matrix for the unlabeled data
[dim1, dim2] = Y_unlabeled.shape
pseudo_Y_unlabeled_train = np.zeros((dim1, dim2))

trainX = np.concatenate((X_labeled, X_unlabeled), axis=0)
trainY = np.concatenate((Y_labeled, pseudo_Y_unlabeled_train), axis=0)


trainset = ds.GeneratorDataset(gen_index_dataset(trainX, trainY), ['image', 'label', 'index'], shuffle=True)
label_train = ds.GeneratorDataset(gen_index_dataset(X_labeled, Y_labeled[:,:-1]), ['image', 'label', 'index'], shuffle=True)

out_dim = Y_labeled.shape[1]

#choice model: linear for UCI dataset, mlp for image dataset
if args.mo == 'linear':
    model = linear_model(X_labeled.shape[1], out_dim)
elif args.mo == 'mlp':
    model = mlp_model(784,500,out_dim)

optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)
if args.uci == 1:
    train_dataloader = trainset.batch(trainX.shape[0])
    # train_dataloader = trainset.batch(args.bs)
    theta = KernelPriorEstimator(X_labeled, X_unlabeled, 5)
else: 
    train_dataloader = trainset.batch(args.bs)
    theta = 0.8
NRPR(train_dataloader, X_test, y_test, model, optimizer, gce_loss, theta, args)
