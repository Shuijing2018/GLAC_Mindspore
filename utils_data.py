import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from copy import deepcopy

def prepare_cv_datasets(dataname):
    train_dataset = dsets.MNIST(root='./dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./dataset/mnist', train=False, transform=transforms.ToTensor())
    full_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset.data), shuffle=True, num_workers=0)
    full_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset.data), shuffle=True, num_workers=0)
    return full_train_loader, full_test_loader

def make_cv_mpu_train_set(full_train_loader, full_test_loader, labeled_num, unlabeled_num, test_num, args):
    for i, (train_data, train_label) in enumerate(full_train_loader):
        print("load_full_train_data!")
    for i, (test_data, test_label) in enumerate(full_test_loader):
        print("load_full_test_data!")
    data = torch.cat((train_data, test_data),0)
    labels = torch.cat((train_label, test_label),0)
    label_set = torch.unique(labels, sorted=True)
    
    random_index = torch.randperm(data.shape[0])
    data = data[random_index]
    labels = labels[random_index]
    print(label_set)
    y_hat = labels.clone()
    labeled_set= [0,1,2,3,4,5]
    unlabeled_set = [6,7,8,9]
    for label in labeled_set:
        is_equal = (labels==label) # true or false
        index = torch.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        
        # select unlabeled, labeled, and test index of each known class 
        y_hat[selected_index[:unlabeled_num]]=-1 # pick the example belonging to the class
        y_hat[selected_index[unlabeled_num:unlabeled_num+labeled_num]] = -2
        y_hat[selected_index[unlabeled_num+labeled_num:unlabeled_num+labeled_num+test_num]] = -3
        
    for label in unlabeled_set:
        is_equal = (labels==label) # true or false
        index = torch.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        
        # select unlabeled, labeled, and test index of each augmented class 
        y_hat[selected_index[:unlabeled_num]]=-1 # pick the example belonging to the class  
        y_hat[selected_index[unlabeled_num:unlabeled_num+test_num]] = -3
    
    ## unlabeled set
    unlabeled_check = (y_hat == -1)
    X_unlabeled_train = data[unlabeled_check]
    y_unlabeled_train = labels[unlabeled_check]
    y_unlabeled_train[:] = unlabeled_set[0]
    
    ### labeled set
    labeled_check = (y_hat == -2)
    X_labeled_train = data[labeled_check]
    y_labeled_train = labels[labeled_check] 
    Y_labeled_train = F.one_hot(y_labeled_train, len(labeled_set)+1)
    Y_unlabeled_train = F.one_hot(y_unlabeled_train, len(labeled_set)+1)
    
    #test set
    test_check = (y_hat == -3)
    X_test = data[test_check]
    y_test = labels[test_check]
    y_test[y_test>=unlabeled_set[0]] = unlabeled_set[0]
    
    print(X_labeled_train.shape, Y_labeled_train.shape, X_unlabeled_train.shape, Y_unlabeled_train.shape, X_test.shape, y_test.shape)
    return X_labeled_train, Y_labeled_train, X_unlabeled_train, Y_unlabeled_train, X_test, y_test


def make_uci_mpu_train_set(ds, num_labeled, num_unlabeled, num_test, seed):
    dataname = './dataset/'+ds+'.mat'
    current_data = sio.loadmat(dataname)
    data = current_data['data']
    Y = current_data['label']
    
    random_index = np.random.permutation(data.shape[0])
    data = data[random_index]
    Y = Y[random_index]
    
    labels = np.argmax(Y, axis=1) # convert one-hot to integer
    if np.min(labels) == 1:
        labels = labels-1
    label_set = np.unique(labels)
    class_prior = np.zeros(label_set.shape)
    y_hat = deepcopy(labels)
    
    for i in range(len(label_set)):
        is_equal = (labels==label_set[i])
        index = np.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        class_prior[i] = selected_index.shape[0]/labels.shape[0]

    label_class_num = int((len(label_set)+1)/2)
    labeled_set = label_set[:label_class_num]
    unlabeled_set = label_set[label_class_num:]
    
    #class prior of know classes in labeled train data
    labeled_prior = class_prior[:label_class_num]/class_prior[:label_class_num].sum()

    
    for label in labeled_set:
        is_equal = (labels==label) # true or false
        index = np.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        
        # number of unlabeled, labeled, and test examples for each known class
        unlabeled_num = np.ceil(num_unlabeled*class_prior[int(label)]).astype(int)
        labeled_num = np.ceil(num_labeled*labeled_prior[int(label)]).astype(int)
        test_num = np.ceil(num_test*class_prior[int(label)]).astype(int)

        # select unlabeled, labeled, and test index of each known class
        y_hat[selected_index[:unlabeled_num]]=-1 # pick the example belonging to the class
        y_hat[selected_index[unlabeled_num:unlabeled_num+labeled_num]] = -2
        y_hat[selected_index[unlabeled_num+labeled_num:unlabeled_num+labeled_num+test_num]] = -3        
        
    for label in unlabeled_set:
        is_equal = (labels==label)
        index = np.arange(0, labels.shape[0])
        selected_index = index[is_equal]
        
        # number of unlabeled, and test examples for each augmented class
        unlabeled_num = np.ceil(num_unlabeled*class_prior[int(label)]).astype(int)
        test_num = np.ceil(num_test*class_prior[int(label)]).astype(int)
   
        # select unlabeled, labeled, and test index of each augmented class
        y_hat[selected_index[:unlabeled_num]]=-1 # pick the example belonging to the class
        y_hat[selected_index[unlabeled_num:unlabeled_num+test_num]] = -3

    # unlabeled set
    unlabeled_check = (y_hat == -1)
    X_unlabeled_train = data[unlabeled_check][:num_unlabeled]
    y_unlabeled_train = labels[unlabeled_check][:num_unlabeled]
    y_unlabeled_train[:] = unlabeled_set[0]
    Y_unlabeled_train = np.eye(len(labeled_set)+1)[y_unlabeled_train].astype(float)
    # Y_unlabeled_train = F.one_hot(y_unlabeled_train, ).float() # conver tinteger  to one-hot

    # labeled set
    labeled_check = (y_hat == -2)
    X_labeled_train = data[labeled_check][:num_labeled]
    y_labeled_train = labels[labeled_check][:num_labeled]
    Y_labeled_train = np.eye(len(labeled_set) + 1)[y_labeled_train].astype(float)
    # Y_labeled_train = F.one_hot(y_labeled_train, len(labeled_set)+1).float() # conver tinteger  to one-hot
    

    #test set
    test_check = (y_hat == -3)
    X_test = data[test_check][:num_test]
    y_test = labels[test_check][:num_test]
    y_test[y_test>=unlabeled_set[0]] = unlabeled_set[0]

    print(X_labeled_train.shape, Y_labeled_train.shape, X_unlabeled_train.shape, Y_unlabeled_train.shape, X_test.shape, y_test.shape)    
    return X_labeled_train, Y_labeled_train, X_unlabeled_train, Y_unlabeled_train, X_test, y_test

class gen_index_dataset:
    def __init__(self, images, given_label_matrix):
        self.images = images
        self.given_label_matrix = given_label_matrix

    def __getitem__(self, index):
        each_image = self.images[index]
        each_label = self.given_label_matrix[index]
        return each_image, each_label, index

    def __len__(self):
        return len(self.given_label_matrix)