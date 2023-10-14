# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 19:04:16 2023

@author: Teng
"""

from MLBOTE_arff import MLBOTE_arff
from multiprocessing import freeze_support
import os
import argparse

def main():   
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='emotions')
    parser.add_argument('--n_labels', '-nl', type=int, default=None)
    parser.add_argument('--alpha_s', '-as', type=float, default=0.01)
    parser.add_argument('--th_c', '-tc', type=float, default=5)
    parser.add_argument('--kb', '-kb', type=int, default=3)
    parser.add_argument('--kw', '-kw', type=int, default=5)
    parser.add_argument('--n_lps', '-lps', type=int, default=10)
    parser.add_argument('--n_jobdist', '-jd', type=int, default=-1)
    parser.add_argument('--n_jobweight', '-jw', type=int, default=2)
    
    args = parser.parse_args()
    
    alpha_s = args.alpha_s
    th_c = args.th_c
    kb = args.kb
    kw = args.kw
    n_lps = args.n_lps
    n_jobs_dist = args.n_jobdist #num of jobs for calculating pair-wise distances
    n_jobs_weight = args.n_jobweight #num of jobs for calculating seed sample weights
    
    label_count_datasheet = {args.dataset: args.n_labels}
    DictdataName = {'emotions': 6, 'scene': 6,'yeast': 14,'medical': 45, 'enron': 53,'genbase': 27,'bibtex': 159,'corel5k':374 }
    if args.dataset in DictdataName:
        if not args.n_labels:
            label_count_datasheet = {args.dataset: DictdataName[args.dataset]}
    if not args.dataset in DictdataName:
        if not args.n_labels:
            raise SystemExit("following arguments are required: --dataset/-d, --n_labels/-nl")
    
    ratio_Disjunction=1            
    
    current_path = os.getcwd()
    dir_datasheet_upper = os.path.join(current_path, "input")
    dir_save_upper = os.path.join(current_path, "output")
    
    MLBOTE_arff(alpha_s, th_c, kb, kw, n_lps, ratio_Disjunction, \
                label_count_datasheet, dir_datasheet_upper,dir_save_upper, n_jobs_dist, n_jobs_weight)
    
if __name__ == '__main__':
    freeze_support()
    main()