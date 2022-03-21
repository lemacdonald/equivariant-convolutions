import torch
import math

class LieGroup():
    def __init__(self, group, device):
        self.group = group
        self.device = device
        if group == 'affine':
            self.dim = 6
            self.basis = torch.zeros(6, 3, 3).to(device)
            self.basis[0] = torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
            self.basis[1] = torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]])
            self.basis[2] = torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]])
            self.basis[3] = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
            self.basis[4] = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
            self.basis[5] = torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]])

            self.ad = torch.zeros(6, 6, 6).to(device)
            for i in range(6):
                for j in range(6):
                    self.ad[i, :, j] = torch.flatten(torch.matmul(self.basis[i], self.basis[j]) - torch.matmul(self.basis[j], self.basis[i]))[:6]

        if group == 'homography':
            self.dim = 8
            self.basis = torch.zeros(8, 3, 3).to(device)
            e1 = torch.tensor([[1., 0., 0.]]).T
            e2 = torch.tensor([[0., 1., 0.]]).T
            e3 = torch.tensor([[0., 0., 1.]]).T
            self.basis[0] = torch.matmul(e1, e1.T) - (1/3)*torch.eye(3)
            self.basis[1] = torch.matmul(e1, e2.T)
            self.basis[2] = torch.matmul(e1, e3.T)
            self.basis[3] = torch.matmul(e2, e1.T)
            self.basis[4] = torch.matmul(e2, e2.T) - (1/3)*torch.eye(3)
            self.basis[5] = torch.matmul(e2, e3.T)
            self.basis[6] = torch.matmul(e3, e1.T)
            self.basis[7] = torch.matmul(e3, e2.T)
            basis = []
            for i in range(8):
                basis.append(self.basis[i].flatten())
            self.basis_mtx = torch.stack(basis).T
            self.basis_pinv = torch.linalg.pinv(self.basis_mtx).to(device)

            self.ad = torch.zeros(8, 8, 8).to(device)
            for i in range(8):
                for j in range(8):
                    self.ad[i, :, j] = self.basis_pinv @ torch.flatten(torch.matmul(self.basis[i], self.basis[j]) - torch.matmul(self.basis[j], self.basis[i]))

    def exp(self, v):
        """
        Computes the exponential of a batch of vectors in the Lie algebra
        """
        vmtx = torch.tensordot(v, self.basis, ([-1], [0]))
        return torch.matrix_exp(vmtx)

    def act(self, u, x):
        """
        Computes the action of a batch of group elements u on a batch of homogeneous coordinates x, as well as the determinant of the Jacobian of the action
        """
        if self.group == 'affine':
            return torch.tensordot(x, u, ([-1], [-1]))[...,:2], torch.det(u).unsqueeze(0).unsqueeze(0).expand(torch.tensor(x.shape[:-1]).tolist() + torch.tensor(u.shape[:-2]).tolist())
        else:
            a = torch.tensordot(x, u, ([-1], [-1]))
            A = torch.zeros(torch.tensor(a.shape[:-1]).tolist() + [2,2]).to(self.device)
            for i in range(2):
                for j in range(2):
                    A[...,i,j] = torch.div(u[...,i,j], a[...,2]) - torch.div(u[...,2,j] * a[...,i], a[...,2] * a[...,2])
            return torch.div(a[...,:2], a[...,2].unsqueeze(-1)), torch.det(A)

    def dvol(self, v, precision=10):
        """
        Computes dvol(v), where v is a batch of Lie algebra elements
        """
        adv = torch.tensordot(v, self.ad, ([-1], [0]))

        e = torch.zeros_like(adv).to(self.device)
        for i in range(precision):
            e = e + (1 / math.factorial(i+1)) * torch.matrix_power( -1. * adv, i)

        return torch.abs(torch.det(e))