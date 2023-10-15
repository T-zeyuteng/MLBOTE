# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:58:47 2020

@author: TENG
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import issparse
from scipy import sparse

SPARSE_FORMAT_TO_CONSTRUCTOR = {
    "bsr": sp.bsr_matrix,
    "coo": sp.coo_matrix,
    "csc": sp.csc_matrix,
    "csr": sp.csr_matrix,
    "dia": sp.dia_matrix,
    "dok": sp.dok_matrix,
    "lil": sp.lil_matrix
}

def get_matrix_in_format(original_matrix, matrix_format):

    if isinstance(original_matrix, np.ndarray):
        return SPARSE_FORMAT_TO_CONSTRUCTOR[matrix_format](original_matrix)

    if original_matrix.getformat() == matrix_format:
        return original_matrix

    return original_matrix.asformat(matrix_format)


def matrix_creation_function_for_format(sparse_format):
    if sparse_format not in SPARSE_FORMAT_TO_CONSTRUCTOR:
        return None

    return SPARSE_FORMAT_TO_CONSTRUCTOR[sparse_format]



def ensure_output_format(matrix, sparse_format='csr', require_dense=False, enforce_sparse=False):

    is_sparse = issparse(matrix)

    if is_sparse:
        if require_dense and not enforce_sparse:
            if matrix.shape[1] != 1:
                return matrix.toarray()
            elif matrix.shape[1] == 1:
                return np.ravel(matrix.toarray())
        else:
            if sparse_format is None:
                return matrix
            else:
                return get_matrix_in_format(matrix, sparse_format)
    else:
        if require_dense and not enforce_sparse:
            # ensuring 1d
            if len(matrix.shape) > 1:
                # a regular dense np.matrix or np.array of np.arrays
                return np.ravel(matrix)
            else:
                return matrix
        else:
            # ensuring 2d
            if len(matrix.shape) == 1:
                matrix = matrix.reshape((matrix.shape[0], 1))
            return matrix_creation_function_for_format(sparse_format)(matrix)
        
def safe_vstack(X_new, y_new, X_resampled, y_resampled):
    if sparse.issparse(X_new):
        X_resampled = sparse.vstack([X_resampled, X_new])
    else:
        X_resampled = np.vstack((X_resampled, X_new))
            
    if sparse.issparse(y_new):
        y_resampled = sparse.vstack([y_resampled, y_new])
        y_resampled = ensure_output_format(
                y_resampled, sparse_format=y_new.getformat(), require_dense=False, enforce_sparse=True)
    else:            
        y_resampled = np.vstack((y_resampled, y_new))
        
        
    
    return X_resampled, y_resampled

def safe_vstack_2(y_new, y_resampled):
            
    if sparse.issparse(y_new):
        y_resampled = sparse.vstack([y_resampled, y_new])
        y_resampled = ensure_output_format(
                y_resampled, sparse_format=y_new.getformat(), require_dense=False, enforce_sparse=True)
    else:            
        y_resampled = np.vstack((y_resampled, y_new))
        
        
    
    return y_resampled