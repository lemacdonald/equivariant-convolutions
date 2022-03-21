import torch
from lie import LieGroup
from data import test_data
from sampling import generate_haar_imgs
import numpy as np
import scipy.io as sio

device = 'cpu'

num_numerals = len(test_data[:][1])

img = torch.zeros(num_numerals, 40, 40).to(device)
label = np.zeros(num_numerals)

for i in range(num_numerals):
    img[i] = test_data[i][0].to(device)
    label[i] = test_data[i][1]

num_samples = 32
scales = [0.15, 0.35, 0.17, 0.35, 0.15, 0.17, 0.15, 0.15]
G = LieGroup('homography', device)

generator = generate_haar_imgs('homography', num_samples, scales, device)

def bilinear_interpolate(im, x, y):

    x0 = torch.floor(x.float()).int()
    x1 = x0 + 1
    y0 = torch.floor(y.float()).int()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1);
    x1 = torch.clamp(x1, 0, im.shape[1]-1);
    y0 = torch.clamp(y0, 0, im.shape[2]-1);
    y1 = torch.clamp(y1, 0, im.shape[2]-1);

    Ia = im[:, x0.long(), y0.long() ]
    Ib = im[:, x0.long(), y1.long() ]
    Ic = im[:, x1.long(), y0.long() ]
    Id = im[:, x1.long(), y1.long() ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def warp(x, u):
    uinv = torch.inverse(u)
    coord_vectors = []
    for i in range(2):
        coord_vectors.append(torch.linspace(-1 , 1 , 40).to(device))
    euc_coords = torch.stack(torch.meshgrid(coord_vectors), 2)
    hom_coords = torch.cat((euc_coords, torch.ones([40,40,1]).to(device)), dim = 2)
    warp_grid, _ = G.act(uinv, hom_coords)
    indices = (20*warp_grid + 20) * (39/ 40)
    T = torch.zeros(x.shape[0], u.shape[0], 40, 40)
    for i in range(40):
        for j in range(40):
            T[...,i,j] = bilinear_interpolate(x, indices[i,j,:,0], indices[i,j,:,1])
    return T

img_data = np.zeros((num_numerals * num_samples, 40, 40))
hom_matrices = np.zeros((num_numerals * num_samples, 3, 3))
labels = np.zeros(num_numerals * num_samples)

for i in range(num_numerals):
    u = generator.generate()
    w = warp(img[i].unsqueeze(0), u).squeeze(0)
    for j in range(num_samples):
        img_data[i*num_samples + j] = w[j].numpy().astype(np.float32)
        hom_matrices[i*num_samples + j] = u[j].numpy().astype(np.float32)
        labels[i*num_samples + j] = label[i]

print(labels.shape)

data_dict = {'img_data': img_data, 'labels': labels, 'hom_matrices': hom_matrices}

filename = 'data/homNIST/homNIST_test'
sio.savemat(filename, data_dict, oned_as='column')


