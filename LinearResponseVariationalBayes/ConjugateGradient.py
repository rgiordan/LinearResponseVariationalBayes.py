# import math

# from autograd import grad, hessian, jacobian, hessian_vector_product
# 
# import autograd.numpy as np
# import autograd.numpy.random as npr

# import copy
import scipy as sp
import numpy as np
from scipy.sparse.linalg import LinearOperator
#from scipy import optimize
#from scipy import stats
import time


# Given a vector of booleans, vec, return two vectors with roughly
# half the number of True values each.
def split_vector(vec):
    split_vec1 = np.full(len(vec), False)
    split_vec2 = np.full(len(vec), False)
    true_inds = np.argwhere(vec)
    num_true = len(true_inds)
    len1 = int(num_true / 2)
    len2 = num_true - len1
    if len1 > 0:
        split_vec1[true_inds[0:len1]] = True
    if len2 > 0:
        split_vec2[true_inds[len1:(len1 + len2)]] = True
    return split_vec1, split_vec2
    

# Recursively split a vector of booleans into smaller vectors until each
# vector has less than terminate_len True values.  Save the results in
# the array results.
def recursive_split(mask, results=[], terminate_len=10):
    if np.sum(mask) > terminate_len:
        # Split and recurse
        mask1, mask2 = split_vector(mask)
        recursive_split(mask1, results=results, terminate_len=terminate_len)
        recursive_split(mask2, results=results, terminate_len=terminate_len)
    else:
        results.append(mask)
        
        
def get_masks(full_len, min_mask_len):
    assert(min_mask_len > 0)
    assert(min_mask_len < full_len)
    masks = []
    ind = 0
    while ind < full_len:
        mask = np.full(full_len, False)
        ind_end = min(ind + min_mask_len, full_len)
        mask[ind:ind_end] = True
        masks.append(mask)
        ind += min_mask_len
    return masks


# A class to solve H^{-1} * vec, where H is the Hessian of the objective
# evaluated at x0.  eval_hessian_vector_product(x0, par) should evaluate to
# H(x0) * par.
class ConjugateGradientSolver(object):
    def __init__(self, eval_hessian_vector_product, x0):
        self.dim = len(x0)
        self.ObjHessVecProdLO = \
            LinearOperator((self.dim, self.dim),
            lambda vec: eval_hessian_vector_product(x0, vec))
        self.x0 = x0
        self.preconditioner = None
        self.tol = 1e-8
        self.initialize()

    def initialize(self):
        self.vecs = []
        self.hinv_vecs = []
        self.masks = []
        self.times = []
        self.cg_infos = []
            
    def get_hinv_vec(self, vec, x0=None):
        hinv_vec, cg_info = sp.sparse.linalg.cg(
            self.ObjHessVecProdLO, vec, x0=x0,
            tol=self.tol, M=self.preconditioner)
        return hinv_vec, cg_info

    # For all subsets of the elements of vec of length no more than
    # terminate_len, store 
    def get_hinv_vec_subsets(self, vec, masks, verbose=False, print_every=10):
        num_masks = len(masks)
        ind = 0
        for mask in masks:
            ind = ind + 1
            if verbose and print_every % ind == 0:
                print('{} of {}\n'.format(ind, num_masks))
            vec_masked = np.zeros(len(vec))
            vec_masked[mask] = vec[mask]
            cg_time = time.time()
            hinv_vec, cg_info = self.get_hinv_vec(vec_masked)
            cg_time = time.time() - cg_time
            self.times.append(cg_time)
            self.vecs.append(vec_masked)
            self.masks.append(mask)
            self.hinv_vecs.append(hinv_vec)
            self.cg_infos.append(cg_info)
            


