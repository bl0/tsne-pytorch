import torch
from torch import nn
import torch.autograd
import numpy as np
import scipy.spatial.distance as distance
from sklearn import manifold

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def tsne(x, dim_embedding=2, perplexity=30):


class TSNE(nn.Module):

    def __init__(self, x, dim_embedding=2, perplexity=30):
        self.n, self.dim = x.shape
        self.dim_embedding = dim_embedding
        self.perplexity = perplexity

        self.x = x
        
        self.y = self.init_y()
        self.p = self.joint_probabilities()

    def init_y(self):
        return torch.normal(torch.zeros(self.n, self.dim_embedding), 1e-4)

    def joint_probabilities(self):
        dist = distance.pdist(self.x, metric='sqeuclidean')
        p = manifold.t_sne._joint_probabilities(dist, self.perplexity, False)
        
        return distance.squareform(p)

    def 


