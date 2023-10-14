# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:09:11 2020

@author: TENG
"""

from skmultilearn.dataset import save_to_arff, load_from_arff
from MLBOTE_class import MLBOTE
import os
from utils import ensure_output_format

def MLBOTE_arff(resample_ra, restresample_th, kb, kw, n_lps, ratio_Disjunction, \
                label_count_datasheet, dir_datasheet_upper,dir_save_upper, n_jobs, n_processes):
    
    
    for datasheet in list(label_count_datasheet.keys()):
        print(datasheet)
        dir_datasheet = os.path.join(dir_datasheet_upper, datasheet, datasheet + '.arff')
        X, y = load_from_arff(dir_datasheet, label_count_datasheet[datasheet])
        
        X = ensure_output_format(X, sparse_format='csr')

        print(datasheet + '-ORSplus_dj')
        '''ORSplus_dj'''
        
        ORS_plusdj = MLBOTE(k1=kb, k2=kb, k3=kw, k4=kw, kr=kw,\
                            n_jobs=n_jobs, n_process=n_processes,\
                                resample_plus=True, adapt_threshold=False,\
                                    resample_disjunction=True)

        
        X_resampled_plusdj, y_resampled_plusdj = ORS_plusdj.fit_resample(
                X,y, resample_ratio = resample_ra, n_lps=n_lps,\
                    restresample_threshold=restresample_th,\
                        ratio_disjunction=ratio_Disjunction)

        
        dir_save = os.path.join(dir_save_upper, datasheet)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
            
        
        dataname = datasheet
        
        label_location = 'end'
        save_sparse = True

        filepath = os.path.join(dir_save, dataname + '_MLBOTE.arff')
        save_to_arff(X_resampled_plusdj, y_resampled_plusdj, label_location=label_location, save_sparse=save_sparse, filename=filepath)

        