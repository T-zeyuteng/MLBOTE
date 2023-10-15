# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:58:00 2020

@author: TENG
"""
import copy
import numpy as np
from scipy import sparse
from utils import ensure_output_format, safe_vstack
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import pandas as pd
from joblib import Parallel, delayed
import os


def err_callback(err):
    print(f'error：{str(err)}')
    
def weight_calcu_singleprocess(label, mgrnsor, self_T_indices, self_R_indices, self_border_T, self_border_R, self_border_T_extend, 
                                self_border_R_extend, self_borderT_nndistance_self, self_borderT_nns_self, 
                                self_borderT_nns_self_forgeneration, self_borderT_nns_self_forgenerationinterior, 
                                self_borderR_nndistance_self, self_borderR_nns_self, self_borderR_nns_self_forgeneration, 
                                self_borderT_nndistance_inR, self_borderT_nns_inR, self_borderR_nndistance_inT, 
                                self_borderR_nns_inT, 
                                lock):
    
    if mgrnsor.batch_count == 0:
        y_indicator_dense = ensure_output_format(
                mgrnsor.y[:,label]==1, require_dense=True, enforce_sparse=False)
        '''target index in the whole data set'''
        target_class_indices = np.flatnonzero(y_indicator_dense)
        
        if len(target_class_indices) == 1:

            self_T_indices[label] = copy.deepcopy(target_class_indices)

            return label, self_T_indices
        
        '''rest index in the whole data set'''
        y_rest_indicator = ~ y_indicator_dense
        rest_class_indices = np.flatnonzero(y_rest_indicator)

        '''target-target，rest-rest，rest-target  distances'''
        X_target_distances = mgrnsor.shared_self_X_distances[np.ix_(
                target_class_indices,target_class_indices)]
        X_rest_distances = mgrnsor.shared_self_X_distances[np.ix_(
                rest_class_indices,rest_class_indices)]
        X_rest_target_distances = mgrnsor.shared_self_X_distances[np.ix_(
                rest_class_indices, target_class_indices)]
        mgrnsor.shared_self_X_distances = None


        k3 = min(mgrnsor.shared_self_k3, len(target_class_indices)-1)
        k1 = min(mgrnsor.shared_self_k1, len(target_class_indices)-1)
        
        nn_rest = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
        nn_rest.fit(X_rest_distances)
        nndistance_rest, nns_rest = nn_rest.kneighbors(n_neighbors=mgrnsor.shared_self_k4, return_distance=True)
        nndistance_target_rest, nns_target_rest = nn_rest.kneighbors(
                X_rest_target_distances.T, n_neighbors=mgrnsor.shared_self_k2,  return_distance=True)
        border_rest = np.unique(nns_target_rest)
        del X_rest_distances

        nn_target = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
        nn_target.fit(X_target_distances)
        nndistance_target, nns_target = nn_target.kneighbors(n_neighbors=k3, return_distance=True)
        
        nndistance_rest_target, nns_rest_target = nn_target.kneighbors(
                X_rest_target_distances[border_rest], n_neighbors=k1, return_distance=True)
        border_target = np.unique(nns_rest_target)  
        del X_rest_target_distances
        
        self_T_indices[label] = target_class_indices
        self_R_indices[label] = rest_class_indices
        
        self_border_T[label] = target_class_indices[border_target]
        self_border_R[label] = rest_class_indices[border_rest]
        self_border_T_extend[label] = target_class_indices[border_target]
        self_border_R_extend[label] = rest_class_indices[border_rest]
        
        self_borderT_nndistance_self[label] = nndistance_target[border_target]
        self_borderT_nns_self[label] = target_class_indices[nns_target[border_target]]
        
        self_borderT_nns_self_forgeneration[label] = target_class_indices[nns_target[border_target]]
        self_borderT_nns_self_forgenerationinterior[label] = target_class_indices[nns_target]
        
        self_borderR_nndistance_self[label] = nndistance_rest[border_rest]
        self_borderR_nns_self[label] = rest_class_indices[nns_rest[border_rest]] 
        del nndistance_rest
        
        self_borderR_nns_self_forgeneration[label] = rest_class_indices[nns_rest[border_rest]] 
        del nns_rest
        
        self_borderT_nndistance_inR[label] = nndistance_target_rest[border_target]
        self_borderT_nns_inR[label] = rest_class_indices[nns_target_rest[border_target]]            
        self_borderR_nndistance_inT[label] = nndistance_rest_target
        self_borderR_nns_inT[label] = target_class_indices[nns_rest_target]
        
    if mgrnsor.batch_count >= 1:
        if label not in self_border_T.keys():
            return
        
        '''new target index in the new data set[True,False,...]'''
        ynew_target_indicator = ensure_output_format(
                mgrnsor.y_new[:,label]==1, require_dense=True, enforce_sparse=False)
        '''new target index in the new data set'''
        new_target_indices = np.flatnonzero(ynew_target_indicator)
        '''new target index in the resampled data set'''
        new_target_idx_inresampled = new_target_indices + mgrnsor.new_begin_idx
        
        '''new rest index in the new data set'''
        ynew_rest_indicator = ~ ynew_target_indicator
        new_rest_indices = np.flatnonzero(ynew_rest_indicator)
        '''new rest index in the resampled data set'''
        new_rest_idx_inresampled = new_rest_indices + mgrnsor.new_begin_idx
        
        new_k3k1_calcu = len(self_T_indices[label]) + len(new_target_idx_inresampled) - 1
        k3 = min(mgrnsor.shared_self_k3, new_k3k1_calcu)
        k1 = min(mgrnsor.shared_self_k1, new_k3k1_calcu)
        
        
        '''new distance matrix'''
        
        '''new_T - all T, new_R - all R, new_R - borderT, new_T - borderR  distances'''
        newT_T_distances = mgrnsor.Xnew_X_distances[np.ix_(
                new_target_indices, np.append(self_T_indices[label],new_target_idx_inresampled) )]

        newR_R_distances = mgrnsor.Xnew_X_distances[np.ix_(
                new_rest_indices, np.append(self_R_indices[label],new_rest_idx_inresampled) )]

        newR_borderT_distances = mgrnsor.Xnew_X_distances[np.ix_(new_rest_indices, self_border_T[label] )]
        
        newT_borderR_distances = mgrnsor.Xnew_X_distances[np.ix_(new_target_indices, self_border_R[label] )]
    

        '''borderT'''
        num_newT, num_T = newT_T_distances.shape
        '''newT KNN in T'''
        if num_newT != 0:
            nn_newT = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
            nn_newT.fit(np.zeros([num_T,num_T]))
            nndistance_newT_, nns_newT = nn_newT.kneighbors(newT_T_distances, n_neighbors=k3 + 1, return_distance=True)
            nndistance_newT_ = nndistance_newT_[:,1:]
            nns_newT = nns_newT[:,1:]

            
            idx_T_resampled = np.append(self_T_indices[label], new_target_idx_inresampled)
            nns_newT = idx_T_resampled[nns_newT] 
                
            '''borderT extend - k3nn、newT distances  '''
            borderT_extend_inTidx = np.zeros(len(self_border_T_extend[label]), dtype=int)
            for i in range(len(borderT_extend_inTidx)):
                borderT_extend_inTidx[i] = np.argwhere(self_T_indices[label] == self_border_T_extend[label][i])
            
            nndistance_borderT_extend = np.hstack((self_borderT_nndistance_self[label], newT_T_distances.T[borderT_extend_inTidx]))
            '''borderT KNN in T'''
            nn_borderT = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
            nn_borderT.fit(np.zeros([min(k3 + num_newT, new_k3k1_calcu), min(k3 + num_newT, new_k3k1_calcu)]))
            nn_distance_borderT, nns_borderT = nn_borderT.kneighbors(nndistance_borderT_extend, n_neighbors=k3, return_distance=True)
            
            
            if self_borderT_nns_self[label].shape[1] < mgrnsor.shared_self_k3:
                borderT_nns_self = np.zeros([self_borderT_nns_self[label].shape[0], k3], dtype=int)
                for i in range(nns_borderT.shape[0]):
                    idx_mapping = np.hstack((self_borderT_nns_self[label][i], new_target_idx_inresampled))
                    borderT_nns_self[i] = copy.deepcopy(idx_mapping[nns_borderT[i]]  )
                self_borderT_nns_self[label] = borderT_nns_self
            else:
                
                for i in range(nns_borderT.shape[0]):
                    idx_mapping = np.hstack((self_borderT_nns_self[label][i], new_target_idx_inresampled))
                    self_borderT_nns_self[label][i] = copy.deepcopy(idx_mapping[nns_borderT[i]]  ) 
                
            '''borderT + newT KNN in T，index in whole'''
            self_borderT_nns_self[label] = np.vstack((self_borderT_nns_self[label], nns_newT))
            self_borderT_nndistance_self[label] = np.vstack((nn_distance_borderT, nndistance_newT_))
            '''borderT + newT index in whole'''
            self_border_T_extend[label] = np.hstack((self_border_T_extend[label], new_target_idx_inresampled))
        
        
        ''' borderR               '''
        num_newR, num_R = newR_R_distances.shape
        '''newR KNN in R'''
        if num_newR != 0:                
            nn_newR = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
            nn_newR.fit(np.zeros([num_R,num_R]))
            nndistance_newR_, nns_newR = nn_newR.kneighbors(newR_R_distances, n_neighbors=mgrnsor.shared_self_k4 + 1, return_distance=True) 
            nndistance_newR_ = nndistance_newR_[:,1:]
            nns_newR = nns_newR[:,1:]

        
            idx_R_resampled = np.append(self_R_indices[label], new_rest_idx_inresampled)
            nns_newR = idx_R_resampled[nns_newR] 

            '''borderR - k4nn、newR distances  '''
            borderR_extend_inRidx = np.zeros(len(self_border_R_extend[label]), dtype=int)
            for i in range(len(borderR_extend_inRidx)):
                borderR_extend_inRidx[i] = np.argwhere(self_R_indices[label] == self_border_R_extend[label][i])
                
            nndistance_borderR_extend = np.hstack((self_borderR_nndistance_self[label], newR_R_distances.T[borderR_extend_inRidx]))
            '''borderR KNN in R'''
            nn_borderR = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
            nn_borderR.fit(np.zeros([mgrnsor.shared_self_k4 + num_newR, mgrnsor.shared_self_k4 + num_newR]))
            nn_distance_borderR, nns_borderR = nn_borderR.kneighbors(nndistance_borderR_extend, n_neighbors=mgrnsor.shared_self_k4, return_distance=True)
            
            for i in range(nns_borderR.shape[0]):
                idx_mapping = np.append(self_borderR_nns_self[label][i], new_rest_idx_inresampled)
                self_borderR_nns_self[label][i] = copy.deepcopy(idx_mapping[nns_borderR[i]]  )        
                
            '''borderR + newR KNN in R，index in whole'''
            self_borderR_nns_self[label] = np.vstack((self_borderR_nns_self[label], nns_newR))
            self_borderR_nndistance_self[label] = np.vstack((nn_distance_borderR, nndistance_newR_)  )
            '''borderR + newR index in whole'''
            self_border_R_extend[label] = np.hstack((self_border_R_extend[label], new_rest_idx_inresampled))
        
        '''update T,R'''      
        self_T_indices[label] = np.hstack((self_T_indices[label], new_target_idx_inresampled))
        self_R_indices[label] = np.hstack((self_R_indices[label], new_rest_idx_inresampled))

        
        borderT_nndistance_inR_extend = np.hstack((self_borderT_nndistance_inR[label], newR_borderT_distances.T))
        nn_newR_T = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
        nn_newR_T.fit(np.zeros([mgrnsor.shared_self_k2 + num_newR, mgrnsor.shared_self_k2 + num_newR]))
        nn_distance_borderT_inR, nns_borderT_inR = nn_newR_T.kneighbors(borderT_nndistance_inR_extend, n_neighbors=mgrnsor.shared_self_k2, return_distance=True)
        
        for i in range(nns_borderT_inR.shape[0]):
            idx_mapping = np.append(self_borderT_nns_inR[label][i], new_rest_idx_inresampled)
            self_borderT_nns_inR[label][i] = copy.deepcopy(idx_mapping[nns_borderT_inR[i]]  )             
        
        self_borderT_nndistance_inR[label] = nn_distance_borderT_inR
        
        
        borderR_nndistance_inT_extend = np.hstack((self_borderR_nndistance_inT[label], newT_borderR_distances.T))
        nn_newT_R = NearestNeighbors(metric='precomputed', n_jobs=mgrnsor.shared_self_n_jobs)
        nn_newT_R.fit(np.zeros([min(k1 + num_newT, len(self_T_indices[label])-1), min(k1 + num_newT, len(self_T_indices[label])-1)]))
        nn_distance_borderR_inT, nns_borderR_inT = nn_newT_R.kneighbors(borderR_nndistance_inT_extend, n_neighbors=k1, return_distance=True)
        
        
        
        
        if self_borderR_nns_inT[label].shape[1] < mgrnsor.shared_self_k1:
            borderR_nns_inT = np.zeros([self_borderR_nns_inT[label].shape[0], k1], dtype=int)
            for i in range(nns_borderR_inT.shape[0]):
                idx_mapping = np.append(self_borderR_nns_inT[label][i], new_target_idx_inresampled)
                borderR_nns_inT[i] = copy.deepcopy(idx_mapping[nns_borderR_inT[i]]  )                
            self_borderR_nns_inT[label] = borderR_nns_inT
        else:
                
        
            for i in range(nns_borderR_inT.shape[0]):
                idx_mapping = np.append(self_borderR_nns_inT[label][i], new_target_idx_inresampled)
                self_borderR_nns_inT[label][i] = copy.deepcopy(idx_mapping[nns_borderR_inT[i]]  )          
        
        self_borderR_nndistance_inT[label] = nn_distance_borderR_inT   
        
    mean_T2T_extend = self_borderT_nndistance_self[label].mean(axis=1)
    mean_R2R_extend = self_borderR_nndistance_self[label].mean(axis=1)          
    
    mean_T2R = self_borderT_nndistance_inR[label].mean(axis=1)
    mean_R2T = self_borderR_nndistance_inT[label].mean(axis=1)
    
    T2T_all0_indicator = mean_T2T_extend == 0
    T2R_all0_indicator = mean_T2R == 0
    R2T_all0_indicator = mean_R2T == 0
    T2T_0_indicator = self_borderT_nndistance_self[label][:,0] == 0
    R2R_0_indicator = self_borderR_nndistance_self[label][:,0] == 0
    
    mean_T2R[T2R_all0_indicator] = np.inf
    mean_R2T[R2T_all0_indicator] = np.inf 
    
    
    nn_borderR1_inT =  self_borderR_nns_inT[label][:,0]
    nn_borderR1_inT_inborderTidx = np.zeros(len(nn_borderR1_inT), dtype=int)
    for i in range(len(nn_borderR1_inT)):
        nn_borderR1_inT_inborderTidx[i] = np.argwhere(self_border_T_extend[label] == nn_borderR1_inT[i])
                
    mean_T2T_extend[T2T_all0_indicator] = np.inf
    mean_R2R_extend[R2R_0_indicator] = 0
    T2T_R1nn_inT = copy.deepcopy(mean_T2T_extend[nn_borderR1_inT_inborderTidx]   )
    borderR_density_ratio = mean_R2R_extend[:self_border_R[label].shape[0]] / T2T_R1nn_inT
    borderR_density_ratio = borderR_density_ratio**2
    
    mean_T2T_extend[T2T_0_indicator] = 0
    weight_T = mean_T2T_extend[:self_border_T[label].shape[0]] / mean_T2R
    
    
    weight_R = mean_R2R_extend[:self_border_R[label].shape[0]] / mean_R2T


    self_weight_borderT = {}
    self_weight_borderR = {}
    self_weight_densityratio_TR = {}
    
    self_weight_borderT[label] = weight_T
    self_weight_borderR[label] = weight_R 
    self_weight_densityratio_TR[label] = borderR_density_ratio 
    
    all_vars_inreturned_tuple = {}
    i=2
    for item in ['self_T_indices', 'self_R_indices', 'self_border_T', 'self_border_R', 'self_border_T_extend', \
        'self_border_R_extend', 'self_borderT_nndistance_self', 'self_borderT_nns_self', \
            'self_borderT_nns_self_forgeneration', 'self_borderT_nns_self_forgenerationinterior', \
                'self_borderR_nndistance_self', 'self_borderR_nns_self', 'self_borderR_nns_self_forgeneration', \
                    'self_borderT_nndistance_inR', 'self_borderT_nns_inR', 'self_borderR_nndistance_inT', \
                        'self_borderR_nns_inT', 'self_weight_borderT', 'self_weight_borderR',\
                            'self_weight_densityratio_TR']:
        
        all_vars_inreturned_tuple[item] = i
        i += 1

    return label, all_vars_inreturned_tuple, self_T_indices, self_R_indices, self_border_T, self_border_R, self_border_T_extend, \
        self_border_R_extend, self_borderT_nndistance_self, self_borderT_nns_self, \
            self_borderT_nns_self_forgeneration, self_borderT_nns_self_forgenerationinterior, \
                self_borderR_nndistance_self, self_borderR_nns_self, self_borderR_nns_self_forgeneration, \
                    self_borderT_nndistance_inR, self_borderT_nns_inR, self_borderR_nndistance_inT, \
                        self_borderR_nns_inT, self_weight_borderT, self_weight_borderR, self_weight_densityratio_TR

class vars_multiprocess_weightcalcu_onlyread():
    def __init__(self,y,X_resampled,y_resampled,X_new,y_new,batch_count,
                 X_distances,k1,k2,k3,k4,new_begin_idx,Xnew_X_distances ):
        self.y = y
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        self.X_new = X_new
        self.y_new = y_new
        self.batch_count = batch_count
        if batch_count == 0:
            self.shared_self_X_distances = X_distances
        elif batch_count > 0:
            self.shared_self_X_distances = None
        self.shared_self_k3 = k3
        self.shared_self_k1 = k1
        self.shared_self_k4 = k4
        self.shared_self_k2 = k2
        self.shared_self_n_jobs = 1
        
        self.new_begin_idx = new_begin_idx
        self.Xnew_X_distances = Xnew_X_distances


class MLBOTE():
    
    
    def __init__(self, kn=5, knd=5, k1=3, k2=3, k3=5, k4=5, kr=5, ovrth=10, Cn=1,\
                 n_jobs=-1, n_process=1,
                 random_state=None, resample_plus=False, adapt_threshold=False,
                 resample_disjunction=False):

        self.kn = kn #for noise detect
        self.knd = knd #for noise detect
        self.k1 = k1 # for finding self-borderline samples --> kb in paper
        self.k2 = k2 # for finding cross-borderline samples --> kb in paper
        self.k3 = k3 # for calculating weights with self-borderline samples kNN --> kw in paper
        self.k4 = k4 # for calculating weights with cross-borderline samples kNN --> kw in paper
        self.kr = kr #for resampling --> kw in paper
        self.Cn = Cn #n_samples generated from self-borderline samples \
                     #in the least frequent class is Cn+1 times that in the most frequent one 
        self.n_jobs = n_jobs
        self.n_process = n_process
        self.random_state = random_state
        self.resample_plus = resample_plus
        self.adapt_threshold = adapt_threshold  # adaptive adjust cross-borderline resampling threshold --> restresample_threshold
        self.resample_disjunction = resample_disjunction
        self.ratio_mean = 1
        self.ovrth = ovrth
        self._clean()

    def _clean(self):
        
        self._label_count = None
        self._N_samples = None
        
        self.unique_combinations_ = {}
        self.reverse_combinations_ = []        
        self.label_subconcepts = {}
        
        self.nsample_labels = []
        self.nsample_lplabel = None
        
        self.class_sampling = []      
        self.N_resample_plus = []
        self.restresample_threshold = []
        self.resample_rest_ratio = []
        self.N_resample_rest_before = []
        self.N_resample_rest_after = []
        self.N_resampled_target_eachlabel = None
        self.N_resampled_rest_eachlabel = None
        
        self.T_indices = {}                #extend
        self.R_indices = {}                #extend
        
        self.border_T = {}
        self.border_R = {}
        self.border_T_extend = {}                #extend all new T
        self.border_R_extend = {}                #extend all new R
        
        self.borderT_nndistance_self = {}                #extend update
        self.borderT_nns_self = {}          #index in the whole data set                #extend update
        self.borderR_nndistance_self = {}                #extend update
        self.borderR_nns_self = {}          #index in the whole data set                #extend update
        
        self.borderT_nndistance_inR = {}                # update
        self.borderT_nns_inR = {}          #index in the whole data set                #update
        self.borderR_nndistance_inT = {}                #update
        self.borderR_nns_inT = {}          #index in the whole data set                #update

        self.weight_borderT = {}                 #update
        self.weight_borderR = {}                     #update
        self.weight_densityratio_TR = {}      # rest density T/R            #update
        
        self.borderT_nns_self_forgeneration = {}
        self.borderT_nns_self_forgenerationinterior = {}
        self.borderR_nns_self_forgeneration = {}
        

        self.weight_T_unique = None                #update
        self.weight_R_unique = None                #update
        self.weight_R_unique_TRratio = None                #update
        self.nns_resamplezone = None     #index in the whole data set
        self.only1point_idx = []
        self.N_new_plus = None
        
    def _fit(self, X, y):
        
        N_samples = y.shape[0]

        X_distances = pairwise_distances(X, metric='euclidean', n_jobs=self.n_jobs)
        
        X_distances_0 = np.argwhere(X_distances==0)
        diagonal_non=X_distances_0[:,0]-X_distances_0[:,1]
        index_distances_0_nondiago = np.unique(X_distances_0[diagonal_non.astype(bool)])
        non_repeat = np.ones(N_samples)
        non_repeat[index_distances_0_nondiago] = 0
        non_repeat = non_repeat.astype(bool)

        self.X_distances = copy.deepcopy(X_distances)
        
        self._N_samples, self._label_count = y.shape

        noise_sample_idx, noise_sample_label = self._noise_detect(y)

        noise_after_idx, noise_after_label = self._noise_detect_bydistance(
                y, noise_sample_idx, noise_sample_label)
        
        for i in range(len(noise_after_label)):
            
            y.data[noise_after_idx[i]].pop(y.rows[noise_after_idx[i]].index(noise_after_label[i]))
            y.rows[noise_after_idx[i]].remove(noise_after_label[i])
        
        self.noise_idx = noise_after_idx
        self.noise_label = noise_after_label
        self._N_samples, self._label_count = y.shape

        lp_labelvector = self.label_stastic(y)
        
        self.class_sampling = self._sampling_class(self.nsample_labels, self._N_samples)
        
        self.weight_calcu(y,None,None,None,None,0)
                
        y_forNsampling = y              
        N_samples = y_forNsampling.shape[0]
        nsample_labels = np.array(y_forNsampling.sum(axis=0))
        ovrIRLbl = (N_samples -nsample_labels)/ nsample_labels
        meanovrIR = ovrIRLbl.mean()
             
        return X, y, lp_labelvector, meanovrIR
    
    def fit_resample(self, X, y, resample_ratio = 1, n_lps = 10, step_size=1,
                     restresample_threshold=2.0, adapt_ratio=0.2, resample_rest_ratio=1, ratio_disjunction=2):
        
        weight_T_unique_alliter = {}
        weight_R_unique_TRratio_alliter = {}
        
        X, y, lp_labelvector, meanovrIR = self._fit(X,y)
        self.N_sampling = min(resample_ratio *  meanovrIR , 10 )* ((self.weight_T_unique>0)*1).sum()
        
        self.resamplezone_calcu(lp_labelvector)
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        
        for i in range(n_lps):
            print(str(i)+"  self borderline resampling")
            weight_T_unique_alliter[i+1] = self.weight_T_unique
            
            X_new, y_new = self.resample(X, y, self.N_sampling/n_lps, step_size)  
            if X_new.shape[0]:
                if self.N_resampled_target_eachlabel is None:
                    self.N_resampled_target_eachlabel = y_new.sum(axis=0)
                else:
                    self.N_resampled_target_eachlabel += y_new.sum(axis=0)
                
                X_resampled, y_resampled = safe_vstack(X_new, y_new, X_resampled, y_resampled)
                self.weight_calcu(None, X_resampled, y_resampled, X_new, y_new,i+1)
                
                weight_R_unique_TRratio_alliter[i+1] = self.weight_R_unique_TRratio
            
            if self.resample_plus:                
                print(str(i)+"  cross borderline resampling")
                self.restresample_threshold.append(restresample_threshold)
                
                X_new_plus, y_new_plus = self.resample_rest(X, y, restresample_threshold,
                                                            resample_rest_ratio, step_size)
                if self.N_new_plus is None:
                    self.N_new_plus = X_new_plus.shape[0]
                else:
                    self.N_new_plus += X_new_plus.shape[0]
                

                if not X_new_plus.shape[0]:
                    self.N_resample_rest_after.append(0)
                
                if i == 0:
                    self.N_resampled_rest_eachlabel = np.matrix([0 for i in range(self._label_count)])
                if X_new_plus.shape[0]:
                    if i == 0:
                        self.N_resampled_rest_eachlabel = y_new_plus.sum(axis=0)
                    if i != 0:
                        if self.N_resampled_rest_eachlabel is None:
                            self.N_resampled_rest_eachlabel = y_new_plus.sum(axis=0)
                        else:
                            self.N_resampled_rest_eachlabel += y_new_plus.sum(axis=0)

                    X_resampled, y_resampled = safe_vstack(X_new_plus, y_new_plus, X_resampled, y_resampled)
                    if i != n_lps-1:
                        self.weight_calcu(None, X_resampled, y_resampled, X_new_plus, y_new_plus, i+1)
                    
                    N_resample_plus = self._N_resampleplus_calcu(restresample_threshold)
                    self.N_resample_rest_after.append(N_resample_plus)
                    
                    if self.adapt_threshold:
                                                
                        if N_resample_plus / self.N_resample_plus[-1] > adapt_ratio:
                            restresample_threshold += 0.1

                        elif N_resample_plus == 0 and restresample_threshold>1.1:
                            restresample_threshold -= 0.1

                            
                elif self.adapt_threshold and restresample_threshold>1.1:
                    restresample_threshold -= 0.1

                restresample_threshold = round(restresample_threshold,1)
                
        if self.resample_disjunction:
            N_samples = y_resampled.shape[0]
            print(" interior resampling")
            X_new_disjunction, y_new_disjunction = self.resample_small_disjunction(
                    X, y, N_samples, step_size, ratio_disjunction)
            
            if y_new_disjunction.shape[0]:            
                X_resampled, y_resampled = safe_vstack(
                        X_new_disjunction, y_new_disjunction, X_resampled, y_resampled)
                
            self.N_disjunction_resampled_eachlabel = y_new_disjunction.sum(axis=0)

        
        return X_resampled, y_resampled
        
            
    def _noise_detect(self, y):
        
        noise_sample_idx = []
        noise_sample_label = []
               
        nn_all = NearestNeighbors(metric='precomputed', n_jobs=self.n_jobs)
        nn_all.fit(self.X_distances)
        nndistance_all, nns_all = nn_all.kneighbors(n_neighbors=self.kn, return_distance=True)
        
        y = ensure_output_format(
            y, sparse_format='lil', require_dense=False, enforce_sparse=True)
        y_row = copy.deepcopy(y.rows)
        
        for i in range(self._N_samples):
            nns = nns_all[i]
            
            for label in y_row[i]:
                
                for j in range(self.kn):
                    if label in y_row[nns][j]:
                        break
                    elif j == self.kn - 1:
                        noise_sample_idx.append(i)
                        noise_sample_label.append(label)
                    
                continue
            continue
        
        return np.array(noise_sample_idx), np.array(noise_sample_label)
    
    def _noise_detect_bydistance(self, y, noise_sample_idx, noise_sample_label):
        
        noise_after_idx = np.array([])
        for label in range(self._label_count):
            y_indicator_dense = ensure_output_format(
                    y[:,label]==1, require_dense=True, enforce_sparse=False)

            target_class_indices = np.flatnonzero(y_indicator_dense)
            
            X_target_distances = self.X_distances[np.ix_(
                    target_class_indices,target_class_indices)]
            
            if len(target_class_indices) <= self.knd + 1:
                continue
                        
            nn_target = NearestNeighbors(metric='precomputed', n_jobs=self.n_jobs)
            nn_target.fit(X_target_distances)
            nndistance_target, nns_target = nn_target.kneighbors(n_neighbors=self.knd, return_distance=True)
            
            noise_candiate_indicator = np.array(noise_sample_label) == label
            noise_candiate_bylabel = noise_sample_idx[noise_candiate_indicator]
            
            noise_idx_local = np.zeros(len(noise_candiate_bylabel), dtype=int)
            for i in range(len(noise_candiate_bylabel)):
                noise_idx_local[i] = np.argwhere(target_class_indices == noise_candiate_bylabel[i])
                
            idx_local = [i for i in range(len(target_class_indices))]    
            for local_idx in noise_idx_local:
                idx_local.remove(local_idx)
            
            all_nns = np.unique(nns_target)
            
            for i in range(len(noise_idx_local)):
                if noise_idx_local[i] in all_nns:
                    idx_local.append(noise_idx_local[i])
            
            distance_max = np.max(nndistance_target[idx_local])
            
            indicator = []
            for i in range(len(noise_idx_local)):
                
                if nndistance_target[noise_idx_local[i]][0] > 1.5 * distance_max:
                    indicator.append(True)
                else:
                    indicator.append(False)
            
            
            noise = noise_candiate_bylabel[indicator]
            
            if ~noise_after_idx.any():
                noise_after_idx = noise
                noise_after_label = np.full(len(noise),label)
            else:        
                noise_after_idx = np.hstack((noise_after_idx, noise))
                noise_after_label = np.hstack((noise_after_label, np.full(len(noise),label)))
        
        return noise_after_idx, noise_after_label
                    
            

    def _sampling_class(self, nsample_labels, N_samples):
        
        IRLbl = N_samples / np.array(nsample_labels)
        meanIR = IRLbl.mean()
        ovrIRLbl = (N_samples - np.array(nsample_labels))/ np.array(nsample_labels)
        
        class_resample = []
        for label in range(len(nsample_labels)):
            if ovrIRLbl[label] > self.ovrth or IRLbl[label] > meanIR*self.ratio_mean:
                if self.nsample_labels[label] > 1:
                    class_resample.append(label)
        
        return class_resample
    
 
    def transform_lp(self, y):

        y = ensure_output_format(
            y, sparse_format='lil', require_dense=False, enforce_sparse=True)

        last_id = 0
        lp_labelvector = []
        for labels_applied in y.rows:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations_:
                self.unique_combinations_[label_string] = last_id
                self.reverse_combinations_.append(labels_applied)
                
                for label in labels_applied:
                    if label not in self.label_subconcepts:
                        self.label_subconcepts[label] = [last_id]
                    else:
                        self.label_subconcepts[label].append(last_id)
                    
                
                last_id += 1

            lp_labelvector.append(self.unique_combinations_[label_string])
            

        return np.array(lp_labelvector)
    
    
    
    def label_stastic(self,y):
        
        lp_labelvector = self.transform_lp(y)
        self.nsample_lplabel = np.unique(lp_labelvector,return_counts=True)
        
        self.nsample_labels = y.sum(axis=0).tolist()[0]
        
        return lp_labelvector

    
    def weight_calcu(self, y,  X_resampled, y_resampled, X_new, y_new, batch_count):
        Xnew_X_distances = None
        new_begin_idx = None
        if batch_count >= 1:
            if X_new.shape[0] == 0:
                return None
            Xnew_X_distances = pairwise_distances(X = X_new, 
                                                  Y = X_resampled, metric='euclidean', n_jobs=self.n_jobs)
            new_begin_idx = y_resampled.shape[0]-y_new.shape[0]

        lock = None

        mgrnsor = vars_multiprocess_weightcalcu_onlyread(
            y, X_resampled, y_resampled, X_new, y_new, batch_count, 
            self.X_distances, self.k1, self.k2, self.k3, self.k4, 
            new_begin_idx, Xnew_X_distances)

        if batch_count == 0:

            num_process = self.n_process
            num_process = min(self._label_count,num_process, os.cpu_count()-1)
            
            multi_return = Parallel(n_jobs=num_process, backend="multiprocessing", verbose=3)\
                (delayed(weight_calcu_singleprocess)\
                  (label,mgrnsor,self.T_indices, self.R_indices, self.border_T,
                        self.border_R,self.border_T_extend,self.border_R_extend,
                        self.borderT_nndistance_self,self.borderT_nns_self,
                        self.borderT_nns_self_forgeneration,
                        self.borderT_nns_self_forgenerationinterior,
                        self.borderR_nndistance_self,self.borderR_nns_self,
                        self.borderR_nns_self_forgeneration,self.borderT_nndistance_inR,
                        self.borderT_nns_inR,self.borderR_nndistance_inT,
                        self.borderR_nns_inT, lock) \
                      for label in range(self._label_count) )
        else:

            num_process = self.n_process
            num_process = min(self._label_count,num_process, os.cpu_count()-1)

            multi_return = Parallel(n_jobs=num_process, verbose=3)\
                (delayed(weight_calcu_singleprocess)\
                  (label,mgrnsor,{label:self.T_indices[label]}, {label:self.R_indices[label]}, {label:self.border_T[label]},
                        {label:self.border_R[label]}, {label:self.border_T_extend[label]}, {label:self.border_R_extend[label]},
                        {label:self.borderT_nndistance_self[label]}, {label:self.borderT_nns_self[label]},
                        None,
                        None,
                        {label:self.borderR_nndistance_self[label]}, {label:self.borderR_nns_self[label]},
                        None, {label:self.borderT_nndistance_inR[label]},
                        {label:self.borderT_nns_inR[label]}, {label:self.borderR_nndistance_inT[label]},
                        {label:self.borderR_nns_inT[label]}, lock) \
                      for label in self.R_indices.keys() )


        for item in multi_return:
            
            label = item[0]
            
            if len(item) == 2:
                self.T_indices[label] = copy.deepcopy(item[1][label])
                continue
            
            idx_multireturn = item[1]
            
            self.T_indices[label] = copy.deepcopy(item[idx_multireturn['self_T_indices']][label])
            self.R_indices[label] = copy.deepcopy(item[idx_multireturn['self_R_indices']][label])
            
            self.border_T[label] = copy.deepcopy(item[idx_multireturn['self_border_T']][label])
            self.border_R[label] = copy.deepcopy(item[idx_multireturn['self_border_R']][label])
            self.border_T_extend[label] = copy.deepcopy(item[idx_multireturn['self_border_T_extend']][label])
            self.border_R_extend[label] = copy.deepcopy(item[idx_multireturn['self_border_R_extend']][label])
            
            self.borderT_nndistance_self[label] = copy.deepcopy(item[idx_multireturn['self_borderT_nndistance_self']][label])
            self.borderT_nns_self[label] = copy.deepcopy(item[idx_multireturn['self_borderT_nns_self']][label])
            
            if batch_count == 0:
                self.borderT_nns_self_forgeneration[label] = copy.deepcopy(item[idx_multireturn['self_borderT_nns_self_forgeneration']][label])
                self.borderT_nns_self_forgenerationinterior[label] = copy.deepcopy(item[idx_multireturn['self_borderT_nns_self_forgenerationinterior']][label])
                self.borderR_nns_self_forgeneration[label] = copy.deepcopy(item[idx_multireturn['self_borderR_nns_self_forgeneration']][label] )
            
            self.borderR_nndistance_self[label] = copy.deepcopy(item[idx_multireturn['self_borderR_nndistance_self']][label])
            self.borderR_nns_self[label] = copy.deepcopy(item[idx_multireturn['self_borderR_nns_self']][label] )  
            
            
            
            self.borderT_nndistance_inR[label] = copy.deepcopy(item[idx_multireturn['self_borderT_nndistance_inR']][label])
            self.borderT_nns_inR[label] = copy.deepcopy(item[idx_multireturn['self_borderT_nns_inR']][label])            
            self.borderR_nndistance_inT[label] = copy.deepcopy(item[idx_multireturn['self_borderR_nndistance_inT']][label])
            self.borderR_nns_inT[label] = copy.deepcopy(item[idx_multireturn['self_borderR_nns_inT']][label])
            
            self.weight_borderT[label] = copy.deepcopy(item[idx_multireturn['self_weight_borderT']][label])
            self.weight_borderR[label] = copy.deepcopy(item[idx_multireturn['self_weight_borderR']][label]) 
            self.weight_densityratio_TR[label] = copy.deepcopy(item[idx_multireturn['self_weight_densityratio_TR']][label])
        

        n_class_sampling_max=0
        n_class_sampling_min=np.inf
        for label in self.class_sampling:            
            if label not in self.border_T.keys():
                continue                        
            n_class_sampling_max = max(n_class_sampling_max,self.nsample_labels[label])
            n_class_sampling_min = min(n_class_sampling_min,self.nsample_labels[label])
            
        w_T_df = pd.DataFrame(np.zeros((self._N_samples, self._label_count)), columns=[i for i in range(self._label_count)])
        for label in self.class_sampling:
            if label not in self.border_T.keys():
                continue     
            large_class_p = 1 + self.Cn*(self.nsample_labels[label] -n_class_sampling_min)/ (n_class_sampling_max -n_class_sampling_min + 0.1)
            w_T_df[label][self.border_T[label]] = self.weight_borderT[label]/self.weight_borderT[label].sum()/large_class_p
        
        self.maxweight_label = w_T_df.idxmax(axis=1)
        
        
            
        w_T = np.zeros(self._N_samples)    
        for label in self.class_sampling:
            
            if label not in self.border_T.keys():
                continue                         
            
            large_class_p = 1 + self.Cn*(self.nsample_labels[label] -n_class_sampling_min)/ (n_class_sampling_max -n_class_sampling_min + 0.1)

            w_T[self.border_T[label]] = np.maximum(
                    w_T[self.border_T[label]], self.weight_borderT[label]/self.weight_borderT[label].sum()/large_class_p)
        
        w_T[np.isnan(w_T)] = 0
        
        weight_T = w_T/w_T.sum()
        if len(self.class_sampling) == 0:
            weight_T = w_T
            self.N_sampling = 0
        self.weight_T_unique = copy.deepcopy(weight_T)
            
        w_R_ratio_df = pd.DataFrame(np.zeros((self._N_samples, self._label_count)), columns=[i for i in range(self._label_count)])
        for label in range(self._label_count):   
            if label not in self.border_T.keys():
                continue     
            
            w_R_ratio_df[label][self.border_R[label]] = self.weight_densityratio_TR[label]
        
        self.maxweightR_label = w_R_ratio_df.idxmax(axis=1)

        w_R = np.zeros(self._N_samples)
        w_R_ratio = np.full(self._N_samples, np.inf)
        
        for label in range(self._label_count):
            
            if label not in self.border_T.keys():
                continue 
            
            w_R[self.border_R[label]] = np.maximum(
                    w_R[self.border_R[label]], self.weight_borderR[label])
            w_R_ratio[self.border_R[label]] = np.minimum(
                    w_R_ratio[self.border_R[label]], self.weight_densityratio_TR[label])
        
        w_R_ratio[w_R_ratio == np.inf] = 0
        
        w_R_ratio[np.isnan(w_R_ratio)] = 0
        w_R[np.isnan(w_R)] = 0
        
        weight_R = w_R/w_R.sum()
        self.weight_R_unique = copy.deepcopy(weight_R)
        self.weight_R_unique_TRratio = copy.deepcopy(w_R_ratio)
        
        return None

    
    def resamplezone_calcu(self, lp_labelvector):

        unique_lp = np.unique(lp_labelvector)
        
        self.nns_resamplezone = np.zeros([self._N_samples, self.kr], dtype=int)
        self.nns_resamplezone_lesssample = {}
        
        for lp_label in unique_lp:
            y_indicator_dense = ensure_output_format(
                    lp_labelvector == lp_label, require_dense=True, enforce_sparse=False)
            '''target index in the whole data set'''
            current_lp_indices = np.flatnonzero(y_indicator_dense)
            X_currentlp_distances = self.X_distances[np.ix_(
                    current_lp_indices,current_lp_indices)]
    
            nn_resamplezone = NearestNeighbors(
                    metric='precomputed', n_jobs=self.n_jobs)
            nn_resamplezone.fit(X_currentlp_distances)
            
            if len(current_lp_indices) > self.kr:
                nns_resamplezone = nn_resamplezone.kneighbors(
                    n_neighbors= self.kr, return_distance=False)
                self.nns_resamplezone[current_lp_indices] = copy.deepcopy(current_lp_indices[nns_resamplezone])
                
            elif len(current_lp_indices) > 1:
                nns_resamplezone = nn_resamplezone.kneighbors(
                    n_neighbors= len(current_lp_indices)-1, return_distance=False)
                for j in range(len(current_lp_indices)):
                    self.nns_resamplezone_lesssample[current_lp_indices[j]] = copy.deepcopy(current_lp_indices[nns_resamplezone[j]])
                    
            else:
                self.only1point_idx.append( current_lp_indices[0])

        return None
    
    def resample(self, X, y, N_resampling, step_size):       
        if N_resampling == 0:
            
            return np.array([]), np.array([])
        
        random_state = check_random_state(self.random_state)
        weight_selection = self.weight_T_unique
        N_resampling = int(round(N_resampling))
        
        sample_seed = random_state.choice(self._N_samples, size = N_resampling,
                                       replace=True, p=weight_selection)

        indicator = []
        for i in range(len(sample_seed)):
            indicator.append((sample_seed[i] not in self.only1point_idx) )

        sample_seed = sample_seed[indicator]
        N_resampling = len(sample_seed)        
        sample_refNeigh = random_state.randint(low=0, high=self.kr, size = N_resampling)
        
        
        
        for idx in list(self.nns_resamplezone_lesssample.keys()):
            seed_idx = np.argwhere(sample_seed==idx)
            sample_refNeigh[seed_idx] = random_state.randint(
                    low=0, high=len(self.nns_resamplezone_lesssample[idx]) , size = 1)
            
        steps = step_size * random_state.uniform(size=N_resampling)     
        
        y_new = copy.deepcopy(y[sample_seed])
        
        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(sample_seed, sample_refNeigh, steps)):
                if X[row].nnz:
                    sample = self._generate_sample(X, row, col, step)
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()

            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [N_resampling, X.shape[1]],
                                      dtype=X.dtype),
                    y_new)
    
    def _generate_sample(self, X, seed, refNeigh, step):
        
        if seed in self.nns_resamplezone_lesssample.keys():
            diff_X = X[seed] - X[self.nns_resamplezone_lesssample[seed][refNeigh]]

            return X[seed] - step * diff_X
        else:
            diff_X = X[seed] - X[self.nns_resamplezone[seed, refNeigh]]

            return X[seed] - step * diff_X
        

    def _generate_sample_only1point(self, X, seed, refNeigh, step):
        
        maxweight_label = self.maxweight_label[seed]
        idx_seedinborderT = np.where(self.border_T[maxweight_label]==seed)[0][0]
        reference_samples = self.borderT_nns_self_forgeneration[maxweight_label][idx_seedinborderT]
        
        if len(reference_samples) < self.k3:
            random_state = check_random_state(self.random_state)
            refNeigh = random_state.randint(low=0, high=len(reference_samples), size = 1)
            
        
        diff_X = X[seed] - X[reference_samples[refNeigh]]
        return X[seed] - step * diff_X
  
        
    def _generate_sample_lessthankrpoint(self, X, seed, refNeigh, step):
        
        maxweight_label = self.maxweight_label[seed]
        idx_seedinborderT = np.where(self.border_T[maxweight_label]==seed)[0][0]
        reference_samples = self.borderT_nns_self_forgeneration[maxweight_label][idx_seedinborderT]
        
        if len(reference_samples) < self.k3:
            random_state = check_random_state(self.random_state)
            refNeigh = random_state.randint(low=0, high=len(reference_samples), size = 1)
        
        diff_X = X[seed] - X[reference_samples[refNeigh]]
        return X[seed] - step * diff_X
    
    
    def _generate_sample_only1point_R(self, X, seed, refNeigh, step):
        
        maxweight_label = self.maxweightR_label[seed]
        idx_seedinborderR = np.where(self.border_R[maxweight_label]==seed)[0][0]
        reference_samples = self.borderR_nns_self_forgeneration[maxweight_label][idx_seedinborderR]
        
        if len(reference_samples) < self.k3:
            random_state = check_random_state(self.random_state)
            refNeigh = random_state.randint(low=0, high=len(reference_samples), size = 1)
            
        
        diff_X = X[seed] - X[reference_samples[refNeigh]]
        return X[seed] - step * diff_X
  
        
    def _generate_sample_lessthankrpoint_R(self, X, seed, refNeigh, step):
        
        maxweight_label = self.maxweightR_label[seed]
        idx_seedinborderR = np.where(self.border_R[maxweight_label]==seed)[0][0]
        reference_samples = self.borderR_nns_self_forgeneration[maxweight_label][idx_seedinborderR]
        
        if len(reference_samples) < self.k3:
            random_state = check_random_state(self.random_state)
            refNeigh = random_state.randint(low=0, high=len(reference_samples), size = 1)
        
        diff_X = X[seed] - X[reference_samples[refNeigh]]
        return X[seed] - step * diff_X
    
    def _generate_sample_interior(self, X,y, seed, refNeigh, step, class_resample):
        

        labels_relevant_ = np.where(y.A[seed]==1)[0]
        labels_relevant = []
        for label in labels_relevant_:
            if label in class_resample:
                labels_relevant.append(label)
        labels_relevant = np.array(labels_relevant)
        random_state = check_random_state(self.random_state)
        selected_label = random_state.choice(labels_relevant)
        
        idx_seedinborderT = np.where(self.T_indices[selected_label]==seed)[0][0]
        reference_samples = self.borderT_nns_self_forgenerationinterior[selected_label][idx_seedinborderT]
        
        if len(reference_samples) < self.k3:
            random_state = check_random_state(self.random_state)
            refNeigh = random_state.randint(low=0, high=len(reference_samples), size = 1)
        
        diff_X = X[seed] - X[reference_samples[refNeigh]]
        return X[seed] - step * diff_X
    
    def resample_rest(self, X, y, threshold, resample_rest_ratio, step_size):
        
        candidate_idx = np.where(self.weight_R_unique_TRratio > threshold)[0]
        
        if len(candidate_idx) == 0:
            
            self.N_resample_plus.append(0)
            
            return np.array([]), np.array([])
        
        candidate = self.weight_R_unique_TRratio[candidate_idx]
        
        
        
        N_resample_weight = candidate - 1 
        N_resample_ = np.minimum(N_resample_weight, 5)        
        N_resample = int(N_resample_.sum())
        
        weight_selection = N_resample_weight / N_resample_weight.sum()
     
        
        random_state = check_random_state(self.random_state)
        sample_seed = random_state.choice(
                candidate_idx, size = N_resample, replace=True, p=weight_selection)
        
        indicator = []
        for i in range(len(sample_seed)):
            indicator.append(sample_seed[i] not in self.only1point_idx)
        
        sample_seed = sample_seed[indicator]
        N_resampling = len(sample_seed)
        
        self.N_resample_plus.append(N_resampling)
        
        sample_refNeigh = random_state.randint(low=0, high=self.kr, size = N_resampling)
        
        for idx in list(self.nns_resamplezone_lesssample.keys()):
            seed_idx = np.argwhere(sample_seed==idx)
            sample_refNeigh[seed_idx] = random_state.randint(
                    low=0, high=len(self.nns_resamplezone_lesssample[idx]) , size = 1)
            
        steps = step_size * random_state.uniform(size=N_resampling)     
        
        y_new = copy.deepcopy(y[sample_seed])

        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(sample_seed, sample_refNeigh, steps)):
                if X[row].nnz:
                    sample = self._generate_sample(X, row, col, step)
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()

            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [N_resampling, X.shape[1]],
                                      dtype=X.dtype),
                    y_new)

    
    def _N_resampleplus_calcu(self, threshold):
        
        candidate_idx = np.where(self.weight_R_unique_TRratio > threshold)[0]
        
        if len(candidate_idx) == 0:
            return 0
        
        candidate = self.weight_R_unique_TRratio[candidate_idx]
        N_resample_weight = candidate/threshold - 1
        N_resample_ = np.minimum(np.floor(N_resample_weight), 5)
        N_resample = int(N_resample_.sum())
        return N_resample
    
    def resample_small_disjunction(self, X, y, N_samples, step_size, ratio_mean):
        
        inner_samples = np.intersect1d(np.flatnonzero(self.weight_R_unique==0),
                       np.flatnonzero(self.weight_T_unique==0))
        
        num_sample_eachlabel = []
        for i in range(self._label_count):
            num_sample_eachlabel.append(len(self.T_indices[i]))
        num_sample_eachlabel = np.array(num_sample_eachlabel)
        
        IRLbl = N_samples / num_sample_eachlabel
        meanIR = IRLbl.mean()
        
        class_resample_ = np.flatnonzero(IRLbl>meanIR*ratio_mean)
        class_resample = []
        for class_resample_label in class_resample_:
            if self.nsample_labels[class_resample_label] > 1:
                class_resample.append(class_resample_label)
        class_resample = np.array(class_resample)
        
        class_median = np.setdiff1d(IRLbl, IRLbl[class_resample_]).max()
        class_median = np.flatnonzero(IRLbl==class_median)
        
        if class_resample.any():
            N_resample_eachlabel = num_sample_eachlabel[class_median][0] - num_sample_eachlabel[class_resample]
            N_resample_eachlabel = pd.Series(data=N_resample_eachlabel, index=class_resample)
            self.N_disjunction_resample_eachlabel = copy.deepcopy(N_resample_eachlabel)
        else:
            self.N_disjunction_resample_eachlabel = [0]*y.shape[1]
        
        for i in range(len(class_resample)):
            N_resampling = N_resample_eachlabel[N_resample_eachlabel>0].sort_values()[0:1]
            if N_resampling.any():
                
                class_resampling = N_resampling.index[0]
                N_resampling = N_resampling.values
                
                samples_seed = np.intersect1d(self.T_indices[class_resampling], inner_samples)
                if ~samples_seed.any():
                    samples_seed = self.T_indices[class_resampling][self.T_indices[class_resampling] < self._N_samples]
                X_new, y_new = self._resample_by_seeds(X, y,samples_seed, N_resampling, step_size ,class_resample)
                            
                num_resampled_eachlabel = np.array(y_new.sum(axis=0))[0]
                
                N_resample_eachlabel = N_resample_eachlabel - num_resampled_eachlabel[N_resample_eachlabel.index.values]
                
                class_resample_diff = np.setdiff1d(N_resample_eachlabel.index.values, class_resampling)
                N_resample_eachlabel = N_resample_eachlabel[class_resample_diff]
           
                if i ==0:
                    X_new_all, y_new_all = X_new, y_new
                else:
                    X_new_all, y_new_all = safe_vstack(X_new, y_new, X_new_all, y_new_all)
                    
        if ~class_resample.any():
            return np.array([]), np.array([])
        else:
            
            return X_new_all, y_new_all
                    
                    
            
    def _resample_by_seeds(self, X, y, samples_seed, N_resampling, step_size, class_resample):
        
        random_state = check_random_state(self.random_state)
        
        sample_seed = random_state.choice(samples_seed, size = N_resampling,
                                          replace=True, p=None)

        
        indicator = []
        for i in range(len(sample_seed)):
            indicator.append(sample_seed[i] not in self.only1point_idx)
        
        sample_seed = sample_seed[indicator]
        N_resampling = len(sample_seed)
        
        sample_refNeigh = random_state.randint(low=0, high=self.kr, size = N_resampling)
        
        for idx in list(self.nns_resamplezone_lesssample.keys()):
            seed_idx = np.argwhere(sample_seed==idx)
            sample_refNeigh[seed_idx] = random_state.randint(
                    low=0, high=len(self.nns_resamplezone_lesssample[idx]) , size = 1)
        
        steps = step_size * random_state.uniform(size=N_resampling)  
        
        y_new = copy.deepcopy(y[sample_seed])

        
        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(sample_seed, sample_refNeigh, steps)):
                if X[row].nnz:
                    sample = self._generate_sample(X, row, col, step)
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()

            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [N_resampling, X.shape[1]],
                                      dtype=X.dtype),
                    y_new)

