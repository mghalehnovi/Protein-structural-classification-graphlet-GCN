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
import re
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

###### related functions to approache 1 to 5
def Nets_Savetoleda():
    system("chmod +x ./list2leda")
    data_directory = os.path.join(ROOT_PATH,'PSN_nets',species,'AllPSN4_LCC')
    files = [file for file in os.listdir(data_directory) if file.endswith('.txt')]
    Directory_save = os.path.join(ROOT_PATH,'PSN_nets_leda',species,'AllPSN4_LCC')
    if not os.path.exists(Directory_save):
          os.makedirs(Directory_save)
    for i in range(len(files)):
        file = files[i]
        system("./list2leda "+data_directory+'/'+file+">>"+Directory_save+'/'+file[:-4]+".gw")

def Extract_Graphlets(Order,Feature_selected,level):
    if Order==True:        
       system("chmod +x ./ncount-ordered")
    else: 
       system("chmod +x ./ncount")
       system("chmod +x ./log-convert")
      
    data_directory_Feat_LR = os.path.join(ROOT_PATH,'Feature_for_LR',species,level)
    if not os.path.exists(data_directory_Feat_LR):
              os.makedirs(data_directory_Feat_LR)
    #data_directory_Feat_GCN = os.path.join(ROOT_PATH,"Feature_for_GCN",species,level)
    #if not os.path.exists(data_directory_Feat_GCN):
         #  os.makedirs(data_directory_Feat_GCN)

    # this is data_directory for reading which pdbs of a desired level
    data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
    level_file = level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt'
    level_list_pdb_read = pd.read_csv(data_directory+'/'+level_file, sep='\t')
    level_list_pdb = level_list_pdb_read['PDB_ID'].values.tolist()
    print 'len desired level is ',len(level_list_pdb)
    # this is data_directory for reading psn nets (leda format)
    data_directory = os.path.join(ROOT_PATH, "PSN_nets_leda",species,'AllPSN4_LCC')
    
    X = [] # X is matrix of features
    for i in range(len(level_list_pdb)):
        file = level_list_pdb[i]+'.gw' #files[i]

        if Feature_selected == "orderedgraphlet-3-4": 
           system("./ncount-ordered "+data_directory+"/"+file+" output.txt")
           system("cut -f2 output.txt | tr '\n' '\t' >>p.txt")  
           with open('p.txt') as f:
                first_line = f.readline()
           first_line = [item for item in first_line.split("\t")]
           first_line = [float(item) for item in first_line[:-1]] # int or float, the last element of first_line is '', here it is removed
           #print len(first_line)
        if Feature_selected == "graphlet-3-4":
           system("./ncount "+data_directory+"/"+file+" output.txt")
           system("cut -f2 output.txt | head -8 | tr '\n' '\t' >>p.txt")
           system("./log-convert p.txt p.log") 
           system("mv p.log p.txt")
           with open('p.txt') as f:
                first_line = f.readline()
           first_line = [item for item in first_line.split("\t")]
           first_line = [float(item) for item in first_line]
        if Feature_selected == "graphlet-3-5":
           system("./ncount "+data_directory+"/"+file+" output.txt")
           system("cut -f2 output.txt | tr '\n' '\t' >>p.txt")
           system("./log-convert p.txt p.log") 
           system("mv p.log p.txt")
           with open('p.txt') as f:
                first_line = f.readline()
           first_line = [item for item in first_line.split("\t")]
           first_line = [float(item) for item in first_line]   
           #print len(first_line)
        if i==0:
           X = first_line
        else:
          X = np.vstack((X,first_line))
        if "output.txt":
            system("rm output.txt")
        if 'p.txt':
            system("rm p.txt")  
        # Save first_line for using as initial feature for GCN method
       # np.save(data_directory_Feat_GCN+'/'+file[:-3]+'-feats.npy',first_line)
    # Save feature matrix for LR classification
    np.savetxt(data_directory_Feat_LR+'/'+level+'_'+Feature_selected+'.txt', X,delimiter='\t',fmt="%s")
    if Order==False:
       system("rm output.txt*")

def Read_Data_for_LR(): 
    data_directory_Feat_LR = os.path.join(ROOT_PATH,"Feature_for_LR",species,level)
    X = np.loadtxt(data_directory_Feat_LR+'/'+level+'_'+Feature_selected+'.txt')
    data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
    level_file = level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt'
    level_list_pdb_read = pd.read_csv(data_directory+'/'+level_file, sep='\t')
    Y = np.array(level_list_pdb_read['calss_label'].values.tolist())
    print 'len y is ',len(np.unique(Y))
    return X, Y

def Read_labels():
    data_directory = os.path.join(ROOT_PATH,'PDB_ID_Cath_Label')
    file = level+'_'+species+'_pdb_cath_label.txt'
    pdb_label = pd.read_csv(data_directory+'/'+file, sep='\t')
   # print pdb_label
    pdb = pdb_label[pdb_label.columns[0]].values.tolist()
    labels = pdb_label[pdb_label.columns[1]].values.tolist()
    Dict = {}
    Dict = dict(zip(pdb,labels))
    return Dict

def Read_Features(dict_pdb_labels,level_all):
    data_directory = os.path.join(ROOT_PATH,'unsup-embedding_PSN',species,level_all,model+'_small_0.000010')
    pdb_list = pd.read_csv('PDB_ID_Cath_Label/'+level+'_Homo_sapiens-3.5.169_pdb_cath_label.txt', sep='\t')
    files = pdb_list['PDB_ID'].values.tolist()
    #print files
    List_labels = []
    List_features = []
    for i in range(len(files)): #
        file = files[i]+'.txt'
        #print file
        key = file[:-4]
        label = dict_pdb_labels[key]
        List_labels = np.append(List_labels,label)
        
        raw_data = open(data_directory+"/"+file)
        data_list = list(raw_data)
        data_ar = np.array(data_list)
        #print data_ar
        data_ar_sub = [re.sub('[^0-9]','',data_ar[j]) for j in range(0,len(data_ar))] # just keep numbers not \n 
        data_ar_int = [int(data_ar_sub[i]) for i in range(0, len(data_ar_sub))] # converting to integer	
        Index = sorted(range(len(data_ar_int)),key=data_ar_int.__getitem__) # get index
        #print Index
        var = str(file)[:str(file).find(".txt")]
        name = var+".npy"
        Features = np.load(data_directory+"/"+name)
     #   print Features.shape
        Features_sorted = Features[Index]
        List_features.append(Features_sorted)
        #List_features.append(Features)
    #print np.array(List_features).shape
    #print List_features[1].shape
    #print List_labels

    # Making all features the same size using interpolation
    # Size_all = []
    # for i in range(0,len(List_features)):
       # Size_all = Size_all+[List_features[i].shape[0]]
    # Min_size_0 = min(Size_all)
    # Min_size_1 = List_features[0].shape[1]
    # for i in range(0, len(List_features)):
       # List_features[i] = transform.resize(List_features[i], (Min_size_0,Min_size_1),mode='reflect',anti_aliasing=True) 
    # #print List_features[1].shape
    # nsamples, nx, ny = np.array(List_features).shape
    # List_features = np.array(List_features).reshape((nsamples,nx*ny))
     # Making all features the same size using sum

    for i in range(len(List_features)): #
       List_features[i] = sum(List_features[i])
       #List_features[i] = List_features[i].max(0)
        #List_features[i] = List_features[i].mean(0)
    #print List_features[1].shape
    #print List_features[0].shape
    #List_features = preprocessing.scale(List_features)
    #List_features = preprocessing.minmax_scale(List_features)
   # Making all features the same size using zeropadding
    # Size_all = []
    # for i in range(0,len(List_features)):
       # Size_all = Size_all+[List_features[i].shape[0]]
    # Min_size_0 = max(Size_all)
    # Min_size_1 = List_features[0].shape[1]
    # Matrix_dist = np.zeros((Min_size_0,Min_size_1))
    # for k in range(len(List_features)):
        # result = np.zeros((Min_size_0,Min_size_1))
        # result[:List_features[k].shape[0],:List_features[k].shape[1]] = List_features[k]
        # List_features[k] = result
    
       # #for i in range(0, len(List_features)):
        # #  List_features[i] = transform.resize(List_features[i], (Min_size_0,Min_size_1),mode='reflect',anti_aliasing=True) 
       # #print List_features[1].shape
    # nsamples, nx, ny = np.array(List_features).shape
    # List_features = np.array(List_features).reshape((nsamples,nx*ny))
     #print np.array(List_features).shape
    return List_labels, List_features

def PSN_Graphlet(Feature_selected,Order):
   # Feature extraction: "Nets_Savetoleda" (just one time) and "Extract_Graphlets" should be run. 
   Extract_Graphlets(Order,Feature_selected,level)
   # Classification
   X, Y = Read_Data_for_LR()
   #print X.shape
   #print Y.shape
   return X, Y
   
def PSN_GCN(level_all):
   # Feature extraction: Running "Orca.py" and "Preprocessing_GCN.py" with different initial features and diffrent models for kernels
   # Classification
   dict_pdb_labels = Read_labels() # dict_pdb_labels is a dict of pdb with its label
   #print dict_pdb_labels
   List_labels, List_features = Read_Features(dict_pdb_labels,level_all)
   return np.array(List_features), np.array(List_labels)


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

   
   #Nets_Savetoleda() # This function converts edgelist to leda format for all of psns, just one time running is enough



# # ######Approache Node2vec: PSN, GCN embedding##################
   # print "Approache 5"
   models = ['n2v'] 
   level_all = "Feature_orca_AllPSN4_LCC_128"
   for level in levels:
       print level
       for model in models:
           print model    
           List_features, List_labels = PSN_GCN(level_all)
           LR_classification(List_features,List_labels, sampling_type=0)
           data_directory_save_gcn = os.path.join(ROOT_PATH,'GCN_embedding_features',species,level_all,level)
           if not os.path.exists(data_directory_save_gcn):
              os.makedirs(data_directory_save_gcn)
           np.savetxt(data_directory_save_gcn+'/'+level+'_'+model+'.txt', List_features,delimiter='\t',fmt="%s") 

 