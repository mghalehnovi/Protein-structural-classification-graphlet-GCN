# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019


import sys
import os
import json
from networkx.readwrite import json_graph
import networkx as nx
from networkx.classes.function import set_node_attributes
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from networkx import read_leda
from os import system
import pandas as pd
import numpy as np
import re
from skimage import transform
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import Counter
from imblearn.over_sampling import SMOTE


def Set_Attr_to_Graph(G,percentage_test,percentage_val):
    Len_G = len(G.nodes())
    Num_test = int(percentage_test*Len_G)
    
    Test_values_False = [True for i in range(0,Len_G)]
    Test_dic_False = {k: v for k, v in zip(list(G.nodes())[:Num_test],Test_values_False[:Num_test] )}
    Test_values_True = [False for i in range(0,Len_G)]
    #print Test_dic_False
    Test_dic_True = {k: v for k, v in zip(list(G.nodes())[Num_test:],Test_values_True[Num_test:] )}
    Dict_val_Final = Test_dic_False.copy()
    Dict_val_Final.update(Test_dic_True)
    nx.set_node_attributes(G, values=Dict_val_Final, name='test')
    
    Num_val = int(percentage_val*Len_G)
    Val_values_False = [True for i in range(0,Len_G)]
    Val_dic_False = {k: v for k, v in zip(list(G.nodes())[Num_test:Num_test+Num_val],Val_values_False[Num_test:Num_test+Num_val] )}
    Val_values_True = [False for i in range(0,Len_G)]
    Val_dic_True = {k: v for k, v in zip(list(G.nodes())[Num_test+Num_val:],Val_values_True[Num_test+Num_val:])}
    Dict_val_Final_val = Val_dic_False.copy()
    Dict_val_Final_val.update(Val_dic_True)

    Val_dic_True = {k: v for k, v in zip(list(G.nodes())[:Num_test],Val_values_True[:Num_test])}
    Dict_val_Final_val.update(Val_dic_True)
    nx.set_node_attributes(G, values=Dict_val_Final_val, name='val')

def Write_Graph_to_JSON(G,label):
    Directory_Save = os.path.join(ROOT_PATH,"PSN_nets_json",species,level_all)
    completeName = os.path.join(Directory_Save)
    if not os.path.exists(completeName):
       os.makedirs(completeName)
    with open(Directory_Save+'/'+label+'-G.json', 'w') as outfile1:
         outfile1.write(json.dumps(json_graph.node_link_data(G)))


def Write_id_map_to_JSON(Dict,label):
    Directory_Save = os.path.join(ROOT_PATH,"PSN_nets_json",species,level_all)
    completeName = os.path.join(Directory_Save)
    if not os.path.exists(completeName):
       os.makedirs(completeName)
    with open(Directory_Save+'/'+label+'-id_map.json', 'w') as outfile1:
         outfile1.write(json.dumps(Dict)) 

#Assign an arbitrary class to each node, here it does not matter, in main code of GCN the class won't be used
def Write_class_map_to_JSON(NodesG,label,NodesG_consecutive_int): 
    Directory_Save = os.path.join(ROOT_PATH,"PSN_nets_json",species,level_all)
    completeName = os.path.join(Directory_Save)
    if not os.path.exists(completeName):
       os.makedirs(completeName)
    NodesGI_val_ar = array(NodesG_consecutive_int) 
    onehot_encoder = OneHotEncoder(sparse=False)
    NodesGI_val_ar = NodesGI_val_ar.reshape(len(NodesGI_val_ar), 1)
    onehot_encoded = onehot_encoder.fit_transform(NodesGI_val_ar)
    onehot_encoded_ls=[[int(onehot_encoded[i][j]) for j in range(0,len(onehot_encoded[0]))]for i in range(0,len(onehot_encoded))]
    #ind_list = [i for i in range(len(NodesGI_val_ar))]
    #random.Random((2)).shuffle(ind_list)
    #onehot_encoded_ls=[onehot_encoded_ls[ind_list[i]] for i in range(len(ind_list))]
    Dict_class = {k: v for k, v in zip(NodesG, onehot_encoded_ls)}
    with open(Directory_Save+'/'+label+'-class_map.json', 'w') as outfile1:
         outfile1.write(json.dumps(Dict_class)) 


def preprocessing(): # Make graph ready for GCN input
    data_directory = os.path.join(ROOT_PATH,"PSN_nets/"+species+"/AllPSN4_LCC")
    pdb_list = pd.read_csv('PDB_ID_Cath_Label/'+level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt', sep='\t')
    files = pdb_list['PDB_ID'].values.tolist()
    for i in range(len(files)):#
        file = files[i]+'.txt'
        print file
        G = nx.read_edgelist(data_directory+"/"+file)
        NodesG = G.nodes()
        print 'len node of G is ',len(NodesG)
        if len(NodesG) !=0:
           Set_Attr_to_Graph(G,.097,.11)
           label = file[:-4]
           NodesG_consecutive_int = range(0,len(NodesG))
           Dict = {k: v for k, v in zip(NodesG, NodesG_consecutive_int)}
          # print '\n'
           #test = nx.get_node_attributes(G,'test')
          # print test
           Write_Graph_to_JSON(G,label) #write json file of graph 
           Write_id_map_to_JSON(Dict,label) # write id map of graph (id of each node is mapped to an integer)
           #Write_class_map_to_JSON(NodesG,label,NodesG_consecutive_int)


def GCN_embedding(Dir_save):
    data_directory = os.path.join(ROOT_PATH, "PSN_nets_json",species,level_all)
    files = [f for f in os.listdir(data_directory) if f.endswith("-G.json")]
    for i in range(len(files)): #len(files)
       file = files[i]
       var = str(file)[:str(file).find("-G")]
       #print(var)
       #system("python -m graphsage.unsupervised_train --train_prefix "+data_directory+"/"+var+" --model "+model+" --max_total_steps 1000 --validate_iter 10 --save_directory Homo_sapiens-3.5.169/PPI-nofeature ")
       system("python -m graphsage.unsupervised_train --train_prefix "+data_directory+"/"+var+" --model "+model+" --max_total_steps 1000 --validate_iter 10 --save_directory Homo_sapiens-3.5.169/"+Dir_save)

   
model = sys.argv[1] #graphsage_mean, gcn, graphsage_seq, graphsage_maxpool, graphsage_meanpool,n2v

if __name__=='__main__':
   ROOT_PATH = os.getcwd()
   #print model
   species = 'Homo_sapiens-3.5.169'
   level_all = 'AllPSN4_LCC'  #"AllPSN4_LCC"
   level = 'C'
   #preprocessing() #this function should run for level_all just one time
   
   ##Note : 'flags.DEFINE_integer' in code "unsupervised_train.py" should change to 1. (folder 'graphsage' --> 'unsupervised_train.py' --> line 47)
   #Dir_save = 'Nofeature_AllPSN4_LCC_128'
   #GCN_embedding(Dir_save)
   
   
   Dir_save = 'Feature_orca_AllPSN4_LCC_128'
   GCN_embedding(Dir_save)
   
  