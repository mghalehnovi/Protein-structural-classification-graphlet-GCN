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
import json
from networkx.readwrite import json_graph
import networkx as nx
from networkx.classes.function import set_node_attributes
#from numpy import array
#from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import*
from sklearn.linear_model import LogisticRegression
from random import shuffle
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
from skimage import transform
import re
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from networkx import read_leda
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

def Nodes_to_integer_Map(species):
    data_directory = os.path.join(ROOT_PATH,"PPI_Clean_MaxCC/")
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
           Directory_Save = os.path.join(ROOT_PATH,"GDD_input/",species)
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
    return Dict

def Extract_cath_label(level, species, Dict_Gene_ID_to_int):
    #print level
    data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
    files =[file for file in os.listdir(data_directory) if file.startswith(level) and file.endswith(".txt")]
    data_pdb_cath_label = pd.read_csv(data_directory+'/'+files[0], sep='\t')
    PDB_IDs_list = data_pdb_cath_label['PDB_ID'].values.tolist()
    #print PDB_IDs_list
    data_directory = os.path.join(ROOT_PATH,'Final_Data_Mapping')
    file = species+"__clean_data.txt"
    Mapping_GeneID_PDB = pd.read_csv(data_directory+'/'+file, sep='\t')
    print len(Mapping_GeneID_PDB)
    Mapping_GeneID_PDB_desired_level = Mapping_GeneID_PDB[Mapping_GeneID_PDB['Cross-reference (PDB)'].isin(PDB_IDs_list)]
    print len(Mapping_GeneID_PDB_desired_level)

    # Dict_Gene_ID_Cath_Label: dict of mapping gene id to cath label
    Dict_Gene_ID_Cath_Label = {}
    # Dict_Gene_ID_PDB_ID: dict of mapping gene id to pdb id , we need pdb id in order to concatenate with features of PSN
    Dict_Gene_ID_PDB_ID = {}
    for pdb in PDB_IDs_list:
        Gene_ID = Mapping_GeneID_PDB_desired_level[Mapping_GeneID_PDB_desired_level['Cross-reference (PDB)']==pdb]['Gene ID']
        #print Gene_ID.values[0] ## str(Gene_ID.values[0]) is gene id 
        Dict_Gene_ID_Cath_Label[str(Gene_ID.values[0])] = data_pdb_cath_label[data_pdb_cath_label['PDB_ID']==pdb]['calss_label'].values[0]
        Dict_Gene_ID_PDB_ID[str(Gene_ID.values[0])] = pdb
    #print Dict_Gene_ID_Cath_Label
    Dict_integer_to_Cath_Label = {}
    Dict_integer_to_PDB_ID = {}	
    for key in Dict_Gene_ID_Cath_Label.keys():
        Dict_integer_to_Cath_Label[Dict_Gene_ID_to_int[key]] = Dict_Gene_ID_Cath_Label[key]
        Dict_integer_to_PDB_ID[Dict_Gene_ID_to_int[key]] = Dict_Gene_ID_PDB_ID[key]
    desired_integers_list = sorted(Dict_integer_to_Cath_Label.keys()) #desired_integers_list should be used for desired level in GDD of PPI
    # we sorted index or integers 
    Labels = [] # Labels is sorted labels for integeres (they are cath labels)
    for key in desired_integers_list:
       Labels = np.append(Labels,Dict_integer_to_Cath_Label[key])
    
    PDB_IDs = [] # Labels is sorted labels for integeres (they are cath labels)
    for key in desired_integers_list:
        PDB_IDs = np.append(PDB_IDs,Dict_integer_to_PDB_ID[key])    
    #print PDB_IDs
    return desired_integers_list, Labels, PDB_IDs

def Extract_GDV_PPI(species,order_num): # Extracting ORCA or GDV for PPI network
    system("g++ -O2 -std=c++11 -o orca.exe orca.cpp")
    data_directory_in = os.path.join(ROOT_PATH,"GDD_input",species)
    data_directory_out = os.path.join(ROOT_PATH,"GDD_out",species)
    data_directory_Feat_GCN = os.path.join(ROOT_PATH,"Feature_for_GCN",species)
    if not os.path.exists(data_directory_out):
       os.makedirs(data_directory_out)
    if not os.path.exists(data_directory_Feat_GCN):
       os.makedirs(data_directory_Feat_GCN)
   
    files = [f for f in os.listdir(data_directory_in) if f.endswith(".txt")]
    file = files[0] # here there is only one file
    system("./orca.exe "+str(order_num)+" "+data_directory_in+'/'+file+" "+data_directory_out+'/'+str(order_num)+'_'+file)
    data = pd.read_csv(data_directory_out+"/"+str(order_num)+'_'+file, header=None, sep=' ')
    vals = data.values
    ##norm = preprocessing.minmax_scale(vals[:,:]) # normailzing the features
    np.save(data_directory_Feat_GCN+'/'+str(order_num)+'_'+file[:-4]+'-feats.npy',vals)
    
def Desired_features_for_LR(species,desired_integers_list): 
    data_directory_Feat_LR = os.path.join(ROOT_PATH,"GDD_out",species)
    X = np.loadtxt(data_directory_Feat_LR+'/'+str(order_num)+'_'+species+'_PPI_clean_maxcc.txt') # X is GDD features for all PSN in one level
    print X.shape 
    X = X[desired_integers_list,:]
    #print X[0,:3]
    print X.shape 
    return X

def PPI_GDV(order_num, Dict_Gene_ID_to_int):
   
   #classification
   desired_integers_list, Labels,PDB_IDs  = Extract_cath_label(level,species,Dict_Gene_ID_to_int) 
   X = Desired_features_for_LR(species,desired_integers_list)
   data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
   level_file = level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt'
   level_list_pdb_read = pd.read_csv(data_directory+'/'+level_file, sep='\t')
   PDB_IDs_list = np.array(level_list_pdb_read['PDB_ID'].values.tolist())

   #print PDB_IDs_list
   # We should sort psn_feature based on pdb_list of each level since PSN features is based on that file
   Index_concat = np.array([], dtype='int')
   for i in range(len(PDB_IDs)):
       Indx = np.where(np.array(PDB_IDs)==PDB_IDs_list[i])[0][0] #we use [0][0] since np.where gives us like (array([238]),)
       Index_concat = np.append(Index_concat, int(Indx))
  #print Index_concat
   
   # we have to sort X and labels based on new indices "Index_concat"
   X_sorted = X[Index_concat,:]
   Labels_sorted = Labels[Index_concat]
   return X_sorted, Labels_sorted

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

def preprocessing(): # Make graph ready for GCN input , we do not need class label 
    data_directory = os.path.join(ROOT_PATH,"PPI_Clean_MaxCC")
    file = 'Homo_sapiens-3.5.169_PPI_clean_maxcc.txt'
    print file
    G = nx.read_edgelist(data_directory+"/"+file)
    NodesG = G.nodes()
    print 'len node of G is ',len(NodesG)
    if len(NodesG) !=0:
       Set_Attr_to_Graph(G,.097,.11)
       label = file[:-4]
       NodesG_consecutive_int = range(0,len(NodesG))
       Dict = {k: v for k, v in zip(NodesG, NodesG_consecutive_int)}
       Write_Graph_to_JSON(G,label) #write json file of graph 
       Write_id_map_to_JSON(Dict,label) # write id map of graph (id of each node is mapped to an integer)

# creating embedding or features for PPI   
def GCN_embedding(Dir_save):
    data_directory = os.path.join(ROOT_PATH, "PSN_nets_json",species,level_all)
    files = [f for f in os.listdir(data_directory) if f.endswith("-G.json")]
    for i in range(len(files)): #len(files)
       file = files[i]
       var = str(file)[:str(file).find("-G")]
       #print(var)
       #system("python -m graphsage.unsupervised_train --train_prefix "+data_directory+"/"+var+" --model "+model+" --max_total_steps 1000 --validate_iter 10 --save_directory Homo_sapiens-3.5.169/PPI-nofeature ")
       system("python -m graphsage.unsupervised_train --train_prefix "+data_directory+"/"+var+" --model "+model+" --max_total_steps 1000 --validate_iter 10 --save_directory Homo_sapiens-3.5.169/PPI_"+Dir_save)

def Desired_features_from_GCN_embedding(species,Dict_Gene_ID_to_int,desired_integers_list): 
   # Since the output of GCN is unsorted we need to extract correct index
   dict_index_keys = [] # dict_index_keys is a list of keys 
   for i in range(len(desired_integers_list)): # we should go in order to keep the order of labels unchangeed
       key = Dict_Gene_ID_to_int.keys()[Dict_Gene_ID_to_int.values().index(desired_integers_list[i])]
       dict_index_keys = np.append(dict_index_keys,key)
   #print len(dict_index_keys)
   data_directory_Feat_LR = os.path.join(ROOT_PATH,"unsup-embedding",species,'PPI_'+Dir_save,model+'_small_0.000010')
   raw_data = pd.read_csv(data_directory_Feat_LR+"/Homo_sapiens-3.5.169_PPI_clean_maxcc.txt",header=None, dtype='str')
   raw_data_index = raw_data[raw_data.columns[0]].values.tolist()
   #print raw_data_index
   Index_GCN = np.array([], 'int')
   for i in range(len(dict_index_keys)):#len(dict_index_keys)
       #print dict_index_keys[i]
      # print raw_data_index.index(dict_index_keys[i])
       Index_GCN = np.append(Index_GCN, raw_data_index.index(dict_index_keys[i])) # the goal is to find the index 
   #print Index_GCN
   X = np.load(data_directory_Feat_LR+'/'+species+'_PPI_clean_maxcc.npy') # X is GDD features for all PSN in one level
   print X.shape 
   X = X[Index_GCN,:]
   #print X[0,:3]
   print 'feature shape is ',X.shape
   return X

def PPI_GCN():
   
   Dict_Gene_ID_to_int = Nodes_to_integer_Map(species)  # Dict_Gene_ID_to_int: dict of mapping gene id to integer
   #print Dict_Gene_ID_to_int
   desired_integers_list, Labels,PDB_IDs  = Extract_cath_label(level,species,Dict_Gene_ID_to_int) 
   X = Desired_features_from_GCN_embedding(species,Dict_Gene_ID_to_int,desired_integers_list)
   
   # Sort X and labels based on the pdb file of each level
   data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
   level_file = level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt'
   level_list_pdb_read = pd.read_csv(data_directory+'/'+level_file, sep='\t')
   PDB_IDs_list = np.array(level_list_pdb_read['PDB_ID'].values.tolist())

   #print PDB_IDs_list
   # We should sort psn_feature based on pdb_list of each level since PSN features is based on that file
   Index_concat = np.array([], dtype='int')
   for i in range(len(PDB_IDs)):
       Indx = np.where(np.array(PDB_IDs)==PDB_IDs_list[i])[0][0] #we use [0][0] since np.where gives us like (array([238]),)
       Index_concat = np.append(Index_concat, int(Indx))
  #print Index_concat
   
   # we have to sort X and labels based on new indices "Index_concat"
   X_sorted = X[Index_concat,:]
   Labels_sorted = Labels[Index_concat]
   
   return X_sorted,  Labels_sorted


def Read_Labels(level):
   data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
   level_file = level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt'
   level_list_pdb_read = pd.read_csv(data_directory+'/'+level_file, sep='\t')
   PDB_IDs_list = np.array(level_list_pdb_read['calss_label'].values.tolist())
   return PDB_IDs_list
###### related functions to classification
def Sampling_SMOTE(dat_train,labs_train):
    sm = SMOTE(ratio='auto',random_state=2)
    dat_train_new, labs_train_new = sm.fit_sample(dat_train, labs_train)
    return dat_train_new, labs_train_new  
 
def LR_classification(X,Y, sampling_type): # sampeling_type ==0: no sampling,  sampeling_type ==1: SMOTE sampling
   num_folds = 10
   model = LogisticRegression()

   N = np.size(X,0)
   ind_list = [i for i in range(N)]
   random.Random((2)).shuffle(ind_list)
   X = X[ind_list,:] #shuffleing the data
   Y = Y[ind_list,]
   
   lab_enc = LabelEncoder() # encode labels (here labels are float, we converted them to categories)
   Y = lab_enc.fit_transform(Y)
   
   Counter_Y = Counter(Y)
   keys = sorted(Counter_Y.keys())
   values = [Counter_Y[key] for key in keys]
   #print values
   values_of_fold_element = [int(values[i]/num_folds) for i in range(0,len(values))]
   #print values_of_fold_element
   IndexF = [np.where(Y==keys[i])for i in range(0,len(keys))]
   Accuracy_list = []
   for k in range(0,num_folds):
       te_inds_init = []
       te_inds_init = [np.append(te_inds_init,IndexF[i][0][k*values_of_fold_element[i]:(k+1)*values_of_fold_element[i]]) for i in range(0, len(keys))]
       te_inds  = [int(val) for sublist in te_inds_init for val in sublist]
       tr_inds = [m for m in range(0, len(Y))if m not in te_inds]
       dat_train = X[tr_inds,:]
       labs_train = Y[tr_inds,]
       dat_test = X[te_inds,:]
       labs_test = Y[te_inds]
       if sampling_type==1:
          dat_train_new, labs_train_new = Sampling_SMOTE(dat_train,labs_train)
       else: 
          dat_train_new = dat_train
          labs_train_new = labs_train
    
       model.fit(dat_train_new, labs_train_new)
       acc_each_fold = model.score(dat_test, labs_test)
       Accuracy_list = np.append(Accuracy_list,acc_each_fold)
   print 'acc is ',np.round(Accuracy_list.mean()*100,2)
   print 'std is ',np.round(Accuracy_list.std()*100,2)

   


if __name__ == '__main__':
   ROOT_PATH = os.getcwd()
   species = 'Homo_sapiens-3.5.169'
   levels = ['C', 'A', 'T','H']
   
########Approache 12 and 13: PPI, GDV (ORCA 4 and 5)##################
   # print "Approache 12"
   # order_num = 4 # App12: order_num = 4, App13: order_num = 5
   # #Extracting features
   # Dict_Gene_ID_to_int = Nodes_to_integer_Map(species)  # Dict_Gene_ID_to_int: dict of mapping gene id to integer
   # #Extract_GDV_PPI(species,order_num) # just one time running is enough to extract ORCA features from PPI network
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # if not os.path.exists(Feature_save_directory):
          # os.makedirs(Feature_save_directory)
       # X_sorted, Labels_sorted = PPI_GDV(order_num,Dict_Gene_ID_to_int)
       # LR_classification(X_sorted,Labels_sorted, sampling_type=0)
       # np.savetxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt',X_sorted,fmt='%s')


########Approache 14: PPI, GCN embedding, initial feature: None##################
## Note : 'flags.DEFINE_integer' in code "unsupervised_train.py" should change to 1. (folder 'graphsage' --> 'unsupervised_train.py' --> line 47)
   # print "Approache 14"
   # level_all = 'PPI'  
   # #preprocessing() #this function should run just one time
   # Dir_save='NoInitialFeature'
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # for model in models:
       # print model
       # GCN_embedding(Dir_save) # One time running for each model is enough
       # for level in levels:
           # print level
           # X_sorted_GCN_none, Labels_sorted_GCN_none = PPI_GCN()
           # print X_sorted_GCN_none.shape
           # # #print Labels_sorted_GCN_none
           # LR_classification(X_sorted_GCN_none,Labels_sorted_GCN_none, sampling_type=0)
           # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
           # if not os.path.exists(Feature_save_directory):
                  # os.makedirs(Feature_save_directory)
           # np.savetxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt',X_sorted_GCN_none,fmt='%s')

# ########Approache 15: PPI, GCN embedding, initial feature: GDV (ORCA4)##################
   # print "Approache 15"
   # level_all = '4_PPI'  
   # Dir_save='GDV4InitialFeature'
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # for model in models:
       # print model
       # GCN_embedding(Dir_save) # One time running for each model is enough
       # for level in levels:
           # print level
           # X_sorted_GCN_none, Labels_sorted_GCN_none = PPI_GCN()
           # print X_sorted_GCN_none.shape
           # # #print Labels_sorted_GCN_none
           # LR_classification(X_sorted_GCN_none,Labels_sorted_GCN_none, sampling_type=0)
           # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
           # if not os.path.exists(Feature_save_directory):
                  # os.makedirs(Feature_save_directory)
           # np.savetxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt',X_sorted_GCN_none,fmt='%s')  
  

# ########Approache 16: PPI, GCN embedding, initial feature: GDV (ORCA5)##################
   # print "Approache 16"
   # level_all = '5_PPI'  
   # Dir_save='GDV5InitialFeature'
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # for model in models:
       # print model
       # GCN_embedding(Dir_save) # One time running for each model is enough
       # for level in levels:
           # print level
           # X_sorted_GCN_none, Labels_sorted_GCN_none = PPI_GCN()
           # print X_sorted_GCN_none.shape
           # # #print Labels_sorted_GCN_none
           # LR_classification(X_sorted_GCN_none,Labels_sorted_GCN_none, sampling_type=0)
           # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
           # if not os.path.exists(Feature_save_directory):
                  # os.makedirs(Feature_save_directory)
           # np.savetxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt',X_sorted_GCN_none,fmt='%s')  

# ########Approache 17: PPI, GDV (ORCA4) + GCN embedding, initial feature: None##################
## Note : 'flags.DEFINE_integer' in code "unsupervised_train.py" should change to 1. (folder 'graphsage' --> 'unsupervised_train.py' --> line 47)

   # print "Approache 17"
   # order_num = 4
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # Dir_save='NoInitialFeature'
   
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       # #print X_GDV.shape
       # Labels = Read_Labels(level)
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)    
       # for model in models:
           # print model    
           # X_GCN = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # #print X_GCN.shape
           # feat_concat = np.column_stack((X_GDV,X_GCN))
           # #print feat_concat.shape
           # LR_classification(feat_concat,Labels, sampling_type=0)
   
# ########Approache 18: PPI, GDV (ORCA4) + GCN embedding, initial feature: GDV (ORCA4)##################
   # print "Approache 18"
   # order_num = 4
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # Dir_save='GDV4InitialFeature'
   
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       # #print X_GDV.shape
       # Labels = Read_Labels(level)
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)    
       # for model in models:
           # print model    
           # X_GCN = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # #print X_GCN.shape
           # feat_concat = np.column_stack((X_GDV,X_GCN))
           # #print feat_concat.shape
           # LR_classification(feat_concat,Labels, sampling_type=0) 
    
# ########Approache 19: PPI, GDV (ORCA5) + GCN embedding, initial feature: None##################
## Note : 'flags.DEFINE_integer' in code "unsupervised_train.py" should change to 1. (folder 'graphsage' --> 'unsupervised_train.py' --> line 47)

   # print "Approache 19"
   # order_num = 5
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # Dir_save='NoInitialFeature'
   
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       # #print X_GDV.shape
       # Labels = Read_Labels(level)
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)    
       # for model in models:
           # print model    
           # X_GCN = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # #print X_GCN.shape
           # feat_concat = np.column_stack((X_GDV,X_GCN))
           # #print feat_concat.shape
           # LR_classification(feat_concat,Labels, sampling_type=0)

# ########Approache 20: PPI, GDV (ORCA5) + GCN embedding, initial feature: GDV (ORCA5)##################
   # print "Approache 20"
   # order_num = 5
   # models = ['graphsage_mean','graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn']
   # Dir_save='GDV5InitialFeature'
   
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       # #print X_GDV.shape
       # Labels = Read_Labels(level)
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)    
       # for model in models:
           # print model    
           # X_GCN = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # #print X_GCN.shape
           # feat_concat = np.column_stack((X_GDV,X_GCN))
           # #print feat_concat.shape
           # LR_classification(feat_concat,Labels, sampling_type=0)