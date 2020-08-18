# check the final image folder
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
# import feather
import numpy as np
import math
import sys
import copy
import h5py
from nltk import flatten
import re
import pandas as pd
from collections import Counter
from itertools import combinations

import torch
import torch.nn.parallel
import torch.utils.data as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, models, transforms

# check training folder nonoverlap
dict_image={'train': {'BULKY': [], 'WRAP': [], 'WT': []}, 'validate': {'BULKY': [], 'WRAP': [], 'WT': []}, 'test': {'BULKY': [], 'WRAP': [], 'WT': []}}
fold='/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/emi_nnt_image/data/'
path_main=fold+'LApops_classify/'
for key_sample in dict_image:
    for key_class in dict_image[key_sample]:
        dict_image[key_sample][key_class]=os.listdir(path_main+key_sample+'/'+key_class+'/')

# no repeat: train vs validate vs test
comb=combinations(dict_image.keys(),2)
for sample_1, sample_2  in list(comb):
    namelist1=set([val for sublist in list(dict_image[sample_1].values()) for val in sublist])
    namelist2=set([val for sublist in list(dict_image[sample_2].values()) for val in sublist])
    intersec=namelist1.intersection(namelist2)
    if len(intersec)>0:
        print(sample_1+' '+sample_2)
        print('bad sample separation')


# no repeat: BULKY vs WRAP vs WT
comb=combinations(dict_image['train'].keys(),2)
collect={'0': [], '1': []}
for sample_1, sample_2  in list(comb):
    collect['0']=[]
    collect['1']=[]
    for separa in dict_image:
        collect['0'].append(dict_image[separa][sample_1])
        collect['1'].append(dict_image[separa][sample_2])
    
    namelist1=set([val for sublist in collect['0'] for val in sublist])
    namelist2=set([val for sublist in collect['1'] for val in sublist])
    intersec=namelist1.intersection(namelist2)
    if len(intersec)>0:
        print(sample_1+' '+sample_2)
        print('bad class separation')
        
# check single class within the target class
path_sing_class=fold+'LApops_classify_singleclass/test/'
for key_class in dict_image['test']:
    localfiles=set(os.listdir(path_sing_class+key_class+'/'+key_class))
    intersec=localfiles.intersection(set(dict_image['test'][key_class]))
    if len(intersec)<len(localfiles):
        print('bad single class construction')

# check intepret with in
path_sing_class=fold+'LApops_classify_intepret/test/'
for key_class in dict_image['test']:
    localfiles=set(os.listdir(path_sing_class+key_class+'/'))
    intersec=localfiles.intersection(set(dict_image['test'][key_class]))
    if len(intersec)<len(localfiles):
        print('bad single class construction')

# check external data
dict_image_ext_test={'BULKY': [], 'WRAP': [], 'WT': []}
path_main="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/emi_nnt_image/data/LApops_new_test/test/"
for key_class in dict_image_ext_test:
        dict_image_ext_test[key_class]=os.listdir(path_main+key_class+'/')

# no repeat: BULKY vs WRAP vs WT
comb=combinations(dict_image_ext_test.keys(),2)
collect={'0': [], '1': []}
for sample_1, sample_2  in list(comb):
    collect['0']=dict_image_ext_test[sample_1]
    collect['1']=dict_image_ext_test[sample_2]
    
    namelist1=set([val for val in collect['0']])
    namelist2=set([val for val in collect['1']])
    intersec=namelist1.intersection(namelist2)
    if len(intersec)>0:
        print(sample_1+' '+sample_2)
        print('bad class separation')

sourcedir="/Users/yuewu/Dropbox (Edison_Lab@UGA)/Projects/Bioinformatics_modeling/emi_nnt_image/data/LApops_new_Raw/"
tab_image_ext_test_comp_list=[]
for batchdir in os.listdir(sourcedir):
    if batchdir=='.DS_Store':
        continue
    
    datatab=pd.read_csv(sourcedir+batchdir+"/Classifications.csv",delimiter=",")
    tab_image_ext_test_comp_list.append(datatab)

classtypes=np.array(['WRAP','WT','BULKY'])
tab_image_ext_test_comp=pd.concat(tab_image_ext_test_comp_list)
tab_image_ext_test_comp[tab_image_ext_test_comp['CLASS'].isin(classtypes)]
# check total size of samples

# check match of each class
for key_class in dict_image_ext_test:
    imglist=tab_image_ext_test_comp[tab_image_ext_test_comp['CLASS']==key_class]['IMG'].tolist()
    set1=set(dict_image_ext_test[key_class])
    set2=set([img+'.tif' for img in imglist])
    inters=set1.intersection(set2)
    if len(inters)<len(set2) | len(inters)<len(set1):
        print('bad external data class construction')
