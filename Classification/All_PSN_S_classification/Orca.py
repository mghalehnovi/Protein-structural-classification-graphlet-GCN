# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import os
import pandas as pd
import sys
import networkx as nx
import numpy as np
import csv
from os import system
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from random import shuffle
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
import warnings
warnings.filterwarnings("ignore")

def Nodes_to_integer_Map(species):
    data_directory = os.path.join(ROOT_PATH,"PSN_nets/"+species+"/"+level_all)
    files = [f for f in os.listdir(data_directory) if f.endswith(".txt")]
    for i in range(0,len(files)):
        file = files[i]
        #print file
        G = nx.read_edgelist(data_directory+"/"+file)
        NodesG = G.nodes()
        #print len(NodesG)
        if len(NodesG) !=0:
           NodesG_consecutive_int = range(0,len(NodesG))
           Dict = {k: v for k, v in zip(NodesG, NodesG_consecutive_int)}
           G = nx.relabel_nodes(G,Dict)
           N = len(G.nodes())
           E = len(G.edges())
           Directory_Save = os.path.join(ROOT_PATH,"GDD_input",species,level_all)
           if not os.path.exists(Directory_Save):
              os.makedirs(Directory_Save)
           nx.write_edgelist(G,Directory_Save+'/'+file,data=False)
           # now we should add number of nodes and edges to the file in order to be proper as input of orca software
           Data = pd.read_csv(Directory_Save+'/'+file, header=None)
           Data.loc[-1] = str(N)+" "+str(E)
           Data.index = Data.index+1
           Data.sort_index(inplace=True)
           Data.to_csv(Directory_Save+'/'+file,header=None, index=False)
        else:
           print file
    
def Extract_GDD(species,level_all,order_num):
    system("g++ -O2 -std=c++11 -o orca.exe orca.cpp")
    data_directory_in = os.path.join(ROOT_PATH,"GDD_input",species,level_all)
    data_directory_out = os.path.join(ROOT_PATH,"GDD_out",species,level_all)
    data_directory_Feat_GCN = os.path.join(ROOT_PATH,"Feature_for_GCN",species,level_all)
    if not os.path.exists(data_directory_out):
       os.makedirs(data_directory_out)
    if not os.path.exists(data_directory_Feat_GCN):
       os.makedirs(data_directory_Feat_GCN)
    
   
    pdb_list = pd.read_csv('PDB_ID_Cath_Label/'+level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt', sep='\t')
    files = pdb_list['PDB_ID'].values.tolist()
 
     
    for i in range(0,len(files)):
        file = files[i]+'.txt'
        system("./orca.exe "+str(order_num)+" "+data_directory_in+'/'+file+" "+data_directory_out+'/'+file)
        data = pd.read_csv(data_directory_out+"/"+file, header=None, sep=' ')
        vals = data.values
        #norm = preprocessing.minmax_scale(vals[:,:]) # normailzing the features
        #np.save(data_directory_Feat_GCN+'/'+file[:-4]+'-feats.npy',norm)	
        np.save(data_directory_Feat_GCN+'/'+file[:-4]+'-feats.npy',vals)
       

if __name__ == '__main__':
   ROOT_PATH = os.getcwd()
   species = 'Homo_sapiens-3.5.169'
   level_all = 'AllPSN4_LCC'
   level = "C"
   order_num = 4
   Nodes_to_integer_Map(species) # this one is needed just one time running , it creted inputs for all 1180 psn not 3CMY
   Extract_GDD(species,level_all,order_num)
  