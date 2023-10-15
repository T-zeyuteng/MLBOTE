This is a python code of Multi-label Borderline Oversampling Technique(MLBOTE), which is a pre-process of the classifier training to handle class imbalance in multi-label learning.
### Dependencies
requiremetns: <br>
* Python(>=3.7) <br>
* NumPy(>=1.17.3) <br>
* Pandas(>=1.0.5)<br>
* SciPy(>=1.5.0)<br>
* Scikit-learn(>=1.0.2)<br>
* Scikit-multilearn(>=0.2.0)<br>
* Joblib(>=1.1.0)<br>
### Example
Put the multi-label data set to be resampled (e.g. emotions.arff) into the folder input. 
The resampled data set (e.g. emotions_MLBOTE.arff) will be saved in the folder output after running MLBOTE_main.py with the following command:
```
python MLBOTE_main.py -d emotions.arff -nl 6
```
Required arguments:
```
-d, data set file name
-nl, number of possible labels in the data set
```
#### Parameters
You can adjust the parameters in MLBOTE using the following arguments: <br>
&emsp;-as,&emsp;self-borderline sample resampling ratio $\alpha_s$<br>
&emsp;-tc,&emsp;cross-borderline sample selection threshold $th_c$ <br>
&emsp;-kb,&emsp;number of nearest neighbors for borderline sample identification $k_b$ <br>
&emsp;-kw,&emsp;number of nearest neighbors for borderline sample selection weights calculation and reference samples selection $k_w$ <br>
&emsp;-lps,&emsp;number of loops in borderline sample resampling  $n_{lps}$ <br>
&emsp;-jd,&emsp;number of processes for pair-wise distances computing <br>
&emsp;-jw,&emsp;number of processes for borderline sample selection weights computing (a large number may cause out-of-memory error) <br>
For example, 
```
python MLBOTE_main.py -d emotions.arff -nl 6 -as 0.01 -tc 5 -kb 3 -kw 5 -lps 10 -jd 32 -jw 32
```
### Citation
If you use MLBOTE in a scientific publication, we would appreciate citations:
```
@article{TENG2024109953,
title = {Multi-label borderline oversampling technique},
journal = {Pattern Recognition},
volume = {145},
pages = {109953},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109953},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323006519},
author = {Zeyu Teng and Peng Cao and Min Huang and Zheming Gao and Xingwei Wang},
}
```
