# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:09:11 2020

@author: TENG
"""

import numpy as np
from skmultilearn.dataset import save_to_arff, load_from_arff
from sklearn.preprocessing import StandardScaler
from MLBOTE_class import MLBOTE
from sklearn.model_selection import KFold
import os
from utils import ensure_output_format, safe_vstack

def MLBOTE_arff(resample_ra, restresample_th, kb, kw, n_lps, ratio_Disjunction, \
                label_count_datasheet, dir_datasheet_upper,dir_save_upper, n_jobs, n_processes):
    
    
    for datasheet in list(label_count_datasheet.keys()):
        print(datasheet)
        dir_datasheet = os.path.join(dir_datasheet_upper, datasheet, datasheet + '.arff')
        if os.path.exists(dir_datasheet):
            X, y = load_from_arff(dir_datasheet, label_count_datasheet[datasheet])
        else:
            dir_datasheet_train = os.path.join(dir_datasheet_upper, datasheet, datasheet + '-train.arff')
            dir_datasheet_test = os.path.join(dir_datasheet_upper, datasheet, datasheet + '-test.arff')
            Xtrain, ytrain = load_from_arff(dir_datasheet_train, label_count_datasheet[datasheet])
            Xtest, ytest = load_from_arff(dir_datasheet_test, label_count_datasheet[datasheet])
            X, y = safe_vstack(Xtest, ytest, Xtrain, ytrain)
            X = ensure_output_format(X, sparse_format='lil', require_dense=False, enforce_sparse=True)
        
        splits_count = 1
        for experiment in range(5):
            kf = KFold(n_splits=2, shuffle=True, random_state=experiment)
            kf.split(X,y)
            
            for train_index, test_index in kf.split(X,y):

                y_train, y_test = y[train_index], y[test_index]
                
                while np.flatnonzero(y_train.sum(axis=0)==0).any():
                    zero_label = np.flatnonzero(y_train.sum(axis=0)==0)[0]
                    movement = np.argwhere(y_test[:,zero_label] == 1)[0][0]
                    train_index = np.append(train_index, test_index[movement])
                    test_index = np.delete(test_index, movement)
                    y_train, y_test = y[train_index], y[test_index]
                    
                while np.flatnonzero(y_test.sum(axis=0)==0).any():            
                    samples0_labels_test = np.flatnonzero(y_test.sum(axis=0)==0)
                    N_intrain = sum(y_train)[:,samples0_labels_test]
                    N1_intrain_indicator = N_intrain == 1
                    only1sample_label = samples0_labels_test[N1_intrain_indicator.A[0]]
                    
                    notonly1_label = samples0_labels_test[~N1_intrain_indicator.A[0]]
                    if ~notonly1_label.any():
                        break
                    idx_intrain = np.flatnonzero((y_train[:,notonly1_label[0]]==1).A)
                    only1_idx = np.argwhere(y_train[:,only1sample_label]==1)[:,0]
                    for idx in idx_intrain:
                        if idx not in only1_idx:
                            test_index = np.append(test_index, train_index[idx])
                            train_index = np.delete(train_index, idx)
                            y_train, y_test = y[train_index], y[test_index]
                            break
                        if idx == idx_intrain[-1]:
                            raise ValueError(" ")
         
                X_train, X_test = X[train_index], X[test_index]
                
            
                scaler = StandardScaler(with_mean=False)
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            

                print(str(splits_count) + '-ORSplus_dj')
                '''ORSplus_dj'''
                
                ORS_plusdj = MLBOTE(k1=kb, k2=kb, k3=kw, k4=kw, kr=kw,\
                                    n_jobs=n_jobs, n_process=n_processes,\
                                        resample_plus=True, adapt_threshold=False,\
                                            resample_disjunction=True)

                
                X_resampled_plusdj, y_resampled_plusdj = ORS_plusdj.fit_resample(
                        X_train,y_train, resample_ratio = resample_ra, n_lps=n_lps,\
                            restresample_threshold=restresample_th[0],\
                                ratio_disjunction=ratio_Disjunction)

                
                dir_save = os.path.join(dir_save_upper, datasheet)
                if not os.path.exists(dir_save):
                    os.makedirs(dir_save)
                    
                
                dataname = datasheet + '_' + str(splits_count)
                
                label_location = 'end'
                save_sparse = True

                filepath = os.path.join(dir_save, dataname + 'train_MLBOTE.arff')
                save_to_arff(X_resampled_plusdj, y_resampled_plusdj, label_location=label_location, save_sparse=save_sparse, filename=filepath)

                filepath = os.path.join(dir_save, dataname + 'train.arff')
                save_to_arff(X_train, y_train, label_location=label_location, save_sparse=save_sparse, filename=filepath)    
                
                filepath = os.path.join(dir_save, dataname + 'test.arff')
                save_to_arff(X_test, y_test, label_location=label_location, save_sparse=save_sparse, filename=filepath)    

                splits_count += 1    
        