import scipy.io as sio
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data.dataset import Subset

path_train = './data/training.mat'  # the padded MNIST training set
path_test = './data/test.mat' # the padded MNIST test set
path_test_aff = './data/affNIST_test.mat' # the affNIST test set
path_test_hom = './data/homNIST_test.mat' # the homNIST test set

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class dataset(data.Dataset):

    def __init__(self, group, data, transform = None, test = None):
        if test:
            if group == 'affine':
                self.labels = data['affNISTdata']['label_int']
                self.img = data['affNISTdata']['image'].transpose().reshape(len(self.labels), 40, 40)
            elif group == 'homography':
                self.labels = data['labels']
                self.img = data['img_data'].astype(np.float32)
        else:
            self.labels = data['affNISTdata']['label_int']
            self.img = data['affNISTdata']['image'].transpose().reshape(len(self.labels), 40, 40)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)
        sample = [img, label]
        return sample

tr_data = loadmat(path_train)
te_data = loadmat(path_test)
te_data_aff = loadmat(path_test_aff)
te_data_hom = loadmat(path_test_hom)

train_data = dataset('affine', tr_data, transform = transforms.ToTensor())
test_data = dataset('affine', te_data, transform=transforms.ToTensor(), test=True)
test_data_aff = dataset('affine', te_data_aff, transform = transforms.ToTensor(), test = True)
test_data_aff_subset = Subset(test_data_aff, list(range(0,10000)))
test_data_hom = dataset('homography', te_data_hom, transform = transforms.ToTensor(), test = True)
test_data_hom_subset = Subset(test_data_hom, list(range(0, 10000)))


