# Author: Mahboobeh Ghalehnovi
# University of Notre Dame, Computer Sceince and Engineering Department
# Date: June 2019

import os
import pandas as pd
import sys
import numpy as np
import csv
from os import system
import random
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from collections import Counter
import random
from skimage import transform
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import*

import warnings
warnings.filterwarnings("ignore")

def Read_Data_for_LR(): 
    data_directory_Feat_LR = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
    X = np.loadtxt(data_directory_Feat_LR+'/'+level+'_'+Feature_selected+'.txt')
    data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
    level_file = level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt'
    level_list_pdb_read = pd.read_csv(data_directory+'/'+level_file, sep='\t')
    Y = np.array(level_list_pdb_read['calss_label'].values.tolist())
    print 'len y is ',len(np.unique(Y))
    return X, Y
	
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

   lab_enc = preprocessing.LabelEncoder() # encode labels (here labels are float, we converted them to categories)
   Y = lab_enc.fit_transform(Y)

   Counter_Y = Counter(Y)
   keys = sorted(Counter_Y.keys())
   values = [Counter_Y[key] for key in keys]
   values_of_fold_element = [int(values[i]/num_folds) for i in range(0,len(values))]
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
   
#########Approache 21: Ordergraphlet3-4 (PSN)+GDV (ORCA4)##################

   # Feature_selected = "orderedgraphlet-3-4" 
   # order_num = 4
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       
       # Y = Read_Labels(level)
       
       # feat_concat = np.column_stack((X_Graphlet,X_GDV))
       # LR_classification(feat_concat,Y, sampling_type=0)

#########Approache 22: graphlet3-4 (PSN)+GDV (ORCA4)##################

   # Feature_selected = "graphlet-3-4" 
   # order_num = 4
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       
       # Y = Read_Labels(level)
       
       # feat_concat = np.column_stack((X_Graphlet,X_GDV))
       # LR_classification(feat_concat,Y, sampling_type=0)

#########Approache 23: graphlet3-5 (PSN)+GDV (ORCA5)##################

   # Feature_selected = "graphlet-3-5" 
   # order_num = 5
   # for level in levels:
       # print level
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GDV_PPI',species,level)
       # X_GDV = np.loadtxt(Feature_save_directory+'/'+str(order_num)+'_GDV.txt')
       
       # Y = Read_Labels(level)
       
       # feat_concat = np.column_stack((X_Graphlet,X_GDV))
       # LR_classification(feat_concat,Y, sampling_type=0)

#########Approache 24: GCN embedding, initial feature: None + GCN embedding, initial feature: None##################
   # print 'Approache 24'
   # models = ["graphsage_mean",'graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn'] 
   # level_all = 'Nofeature_AllPSN4_LCC_128'
   # Dir_save = 'NoInitialFeature'
   # for level in levels: 
       # print level
       # data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
       # for model in models:
           # print model
           # X_GCN_PSN = np.loadtxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt')
           # X_GCN_PPI = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # Y = Read_Labels(level)
           # feat_concat = np.column_stack((X_GCN_PSN,X_GCN_PPI))
           # LR_classification(feat_concat,Y, sampling_type=0)
		   
#########Approache 25: GCN embedding, initial feature: GDV4-matrix + GCN embedding, initial feature: GDV4##################
   # print 'Approache 25'
   # models = ["graphsage_mean",'graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn'] 
   # level_all = 'Feature_orca_AllPSN4_LCC_128'
   # Dir_save = 'GDV4InitialFeature'
   # for level in levels: 
       # print level
       # data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
       # for model in models:
           # print model
           # X_GCN_PSN = np.loadtxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt')
           # X_GCN_PPI = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # Y = Read_Labels(level)
           # feat_concat = np.column_stack((X_GCN_PSN,X_GCN_PPI))
           # LR_classification(feat_concat,Y, sampling_type=0)

#########Approache 26: OrderGraphlet3-4+ GCN embedding, initial feature: None + GCN embedding, initial feature: None##################
   # print 'Approache 26'
   # models = ["graphsage_mean",'graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn'] 
   # level_all = 'Nofeature_AllPSN4_LCC_128'
   # Dir_save = 'NoInitialFeature'
   # Feature_selected = "orderedgraphlet-3-4" 
   # for level in levels: 
       # print level
       # data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
       # for model in models:
           # print model
           # X_GCN_PSN = np.loadtxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt')
           # X_GCN_PPI = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # Y = Read_Labels(level)
           # feat_concat = np.column_stack((X_Graphlet,X_GCN_PSN,X_GCN_PPI))
           # LR_classification(feat_concat,Y, sampling_type=0) 
    
#########Approache 27: OrderGraphlet3-4+ GCN embedding, initial feature: GDV4-matrix + GCN embedding, initial feature: GDV4##################
   # print 'Approache 27'
   # models = ["graphsage_mean",'graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn'] 
   # level_all = 'Feature_orca_AllPSN4_LCC_128'
   # Dir_save = 'GDV4InitialFeature'
   # Feature_selected = "orderedgraphlet-3-4" 
   # for level in levels: 
       # print level
       # data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
       # for model in models:
           # print model
           # X_GCN_PSN = np.loadtxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt')
           # X_GCN_PPI = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # Y = Read_Labels(level)
           # feat_concat = np.column_stack((X_Graphlet,X_GCN_PSN,X_GCN_PPI))
           # LR_classification(feat_concat,Y, sampling_type=0) 

#########Approache 28: Graphlet3-4+ GCN embedding, initial feature: None + GCN embedding, initial feature: None##################
   # print 'Approache 28'
   # models = ["graphsage_mean",'graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn'] 
   # level_all = 'Nofeature_AllPSN4_LCC_128'
   # Dir_save = 'NoInitialFeature'
   # Feature_selected = "graphlet-3-4" 
   # for level in levels: 
       # print level
       # data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
       # for model in models:
           # print model
           # X_GCN_PSN = np.loadtxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt')
           # X_GCN_PPI = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # Y = Read_Labels(level)
           # feat_concat = np.column_stack((X_Graphlet,X_GCN_PSN,X_GCN_PPI))
           # LR_classification(feat_concat,Y, sampling_type=0)

#########Approache 29: OrderGraphlet3-4+ GCN embedding, initial feature: GDV4-matrix + GCN embedding, initial feature: GDV4##################
   # print 'Approache 29'
   # models = ["graphsage_mean",'graphsage_maxpool', 'graphsage_meanpool','graphsage_seq','gcn'] 
   # level_all = 'Feature_orca_AllPSN4_LCC_128'
   # Dir_save = 'GDV4InitialFeature'
   # Feature_selected = "graphlet-3-4" 
   # for level in levels: 
       # print level
       # data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
       # Feature_save_directory = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
       # X_Graphlet = np.loadtxt(Feature_save_directory+'/'+level+'_'+Feature_selected+'.txt')
       # Feature_save_directory = os.path.join(ROOT_PATH,'Feature_save_GCN_PPI',species,Dir_save,level)
       # for model in models:
           # print model
           # X_GCN_PSN = np.loadtxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt')
           # X_GCN_PPI = np.loadtxt(Feature_save_directory+'/'+level+'_'+model+'_GCN.txt')
           # Y = Read_Labels(level)
           # feat_concat = np.column_stack((X_Graphlet,X_GCN_PSN,X_GCN_PPI))
           # LR_classification(feat_concat,Y, sampling_type=0) 