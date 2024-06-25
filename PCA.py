import pickle as pkl
import numpy as np

def pca(X: np.array, k: int) -> np.array:
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    Return (a*b,k) np array comprising the k normalised basis vectors comprising the k-dimensional subspace for all images
    where the first column must be the most relevant principal component and so on
    """
    # TODO
    # pass
    X = X.reshape(X.shape[0],-1)
    # X = X.T
    # X = X - np.mean(X,axis = 1)
    mean_X = np.mean(X,axis = 0)
    covar = np.cov(X-mean_X,rowvar = False)
    eig_val,eig_vec = np.linalg.eigh(covar)
    order = np.argsort(eig_val)[::-1]
    sort_vec = eig_vec[:,order]
    return sort_vec[:,:k]
    #END TODO
    

def projection(X: np.array, basis: np.array):
    """
    X is an (N,a,b) array comprising N images, each of size (a,b).
    basis is an (a*b,k) array comprising of k normalised vectors
    Return (n,k) np array comprising the k dimensional projections of the N images on the normalised basis vectors
    """
    # TODO
    # pass
    X1 = X.reshape((X.shape[0],-1))
    projections = np.matmul(X1,basis)
    return projections
    # END TODO
    