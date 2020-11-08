import tensorflow as tf
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow.keras
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import networkx as nx
import random
import matplotlib.pyplot as plt
import sklearn
import random
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn import preprocessing
import scipy as sc
import os
import re
import gc
import itertools
import statistics
import pickle
import argparse
import argparse
import random
from numpy.random import seed
import os
def glrt_init(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def GCN_layer(A1,H,out_feat,in_feat,act,name='gcn',i=0,k=1,train=True):
    weights = glrt_init([in_feat, out_feat],name=name)
    n12=A1.get_shape().as_list()
    eps=tf.Variable(tf.zeros(1))
    rowsum = tf.reduce_sum(A1,axis=2)            
    d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
    d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)        
    d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
    A1=tf.matmul(tf.matmul(d_inv_sqrt,A1),d_inv_sqrt)
    H = tf.nn.dropout(H, 1-k)
    A1=tf.matmul(A1,H,name=name+'matmul1')
    A1=A1+(1+eps)*H 
    for i in range(2-1):
      ad=tf.keras.layers.Dense(units = out_feat)(A1)
      ab=tf.keras.layers.BatchNormalization()(ad)
      input_features = tf.nn.relu(ad)

    H_next=tf.keras.layers.Dense(units = out_feat)(A1)#(input_features)
    ab=tf.keras.layers.BatchNormalization()(H_next)
    H_next = tf.nn.relu(ab)
    return H_next


class hierarchical_repr(object):
    def __init__(self,placeholders):
        self.num_pool=placeholders['num_pool']
        self.num_nodes=placeholders['num_nodes']
        self.feat_dim=placeholders['feat_dim']
        self.feat_dim1=placeholders['feat_dim1']
        self.embd_dim=placeholders['emb_dim']
        self.gcn_dim=placeholders['gcn_dim']
        self.clusratio=placeholders['clusrstio']
        self.nclasses=placeholders['nclasses']
        self.learning_rate=placeholders['learning_rate']
        self.hidden_size=placeholders['emb_dim']
        self.acts="relu"
        self.ran=placeholders['ran']
        self.ran1=placeholders['ran']
        self.batch_size=placeholders['batch_size']
        

    def bilinear(self,s,X_comb):
        initializer = tf.keras.initializers.he_normal
        shape1=X_comb.get_shape().as_list()
        w_orig = tf.Variable(tf.random_normal([shape1[-1], shape1[-1]], stddev=0.1))
        out_corr=[]        
        out_orig=tf.tensordot(X_comb,w_orig,axes=[[-1],[0]])
        s_T=tf.transpose(s,[0,2,1])
        c=tf.reshape(s_T,[-1,s_T.get_shape().as_list()[1],1])
        out_orig1=tf.matmul(out_orig,c)
        out_orig2=tf.reshape(out_orig1,[-1,X_comb.get_shape().as_list()[1]])
        return out_orig2 
    

    def bilinear1(self,s,X_comb):
        initializer = tf.keras.initializers.he_normal
        shape1=X_comb.get_shape().as_list()
        w_orig = tf.Variable(tf.random_normal([shape1[-1], s.get_shape().as_list()[2]], stddev=0.1))
        out_corr=[]     
        out_orig=tf.tensordot(X_comb,w_orig,axes=[[-1],[0]])
        s_T=tf.transpose(s,[0,2,1])
        c=tf.reshape(s_T,[-1,s_T.get_shape().as_list()[1],1])
        out_orig1=tf.matmul(out_orig,c)#,axes=[[-1],[0]])
        out_orig2=tf.reshape(out_orig1,[-1,X_comb.get_shape().as_list()[1]])
        return out_orig2 


    def Emb_Pooling_layer(self,clusnext,A3,x,in_feat,act,j,i):  
        if clusnext==-1:
            if j==1:
                clusnext1=1            
            else:
                if i==0:
                    clusnext1=4
                elif i==1:
                    clusnext1=2
        else:
            clusnext1=clusnext
        
        with tf.variable_scope("node_gnn",reuse=False):
            x1=x
            A2=A3
            for i1 in range(1):
                z_l=GCN_layer(A2,x1,self.gcn_dim,in_feat,act,i=j,train=self.train,k=self.drop_rate)
                x1=z_l
                in_feat=x1.get_shape().as_list()[2] 
            x1=x
            in_feat=x1.get_shape().as_list()[2] 
            for i1 in range(1):                
                z_l1=GCN_layer(A2,x1,clusnext1,in_feat,act,i=j,train=self.train,k=self.drop_rate)
                x1=z_l1
                in_feat=x1.get_shape().as_list()[2]
                
            s_l=tf.nn.softmax(z_l1)
            x_l1=tf.matmul(tf.transpose(s_l,[0,2,1]),z_l)  
            A_l1=tf.matmul(tf.matmul(tf.transpose(s_l,[0,2,1]),A2),s_l)       
        return x_l1,A_l1,clusnext1
    
    
    def Calc_Loss(self, lab, lab_orig,lab1,lab_local,reuse=False):
        l2_reg_lambda=1e-5
        lab = tf.reshape(lab, [-1])
        lab1 = tf.reshape(lab1, [-1])
        lab_local = tf.reshape(lab_local, [-1])
        lab_orig = tf.reshape(lab_orig, [-1])
        lab_orig1=tf.nn.sigmoid(lab_orig)
        print('lab_orig',lab_orig1)
        with tf.name_scope('loss'):
            lossL2=0

            self.loss_val = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=lab_orig, labels=lab))
            self.loss_val1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=lab_local, labels=lab1))  
        self.optimizer()
    
    def hierarchical_repr_arch(self):
        
        ##############COMPLETE ARCHITECHTURE OF THE PAPER################
        self._add_placeholders()
        with tf.variable_scope("node12_gnn",reuse=False):           
            A=self.input_adj  
            x=self.input_features
            in_feat=x.get_shape().as_list()            
            in_feat=in_feat[2]
            tot_nodes=0
            s=[]

            if self.clusratio!=-1:
                clusnext=int(self.num_nodes*self.clusratio)
            else:
                clusnext=-1
            x=GCN_layer(A,x,self.gcn_dim,in_feat,"relu",i=3,train=self.train,k=self.drop_rate)
            in_feat=x.get_shape().as_list()            
            in_feat=in_feat[2]
            
            for i in range(self.num_pool): 
                if i==self.num_pool-1:
                    x_l1,A_l1,lab=self.Emb_Pooling_layer(1,A,x,x.get_shape().as_list()[2],self.acts,1,i)
                    s=x_l1
                    self.embeddings=s
                    
                else:                    
                    x_l1,A_l1,lab=self.Emb_Pooling_layer(clusnext,A,x,x.get_shape().as_list()[2],self.acts,0,i)
                    if i!=0.:
                        X_comb=tf.concat([X_comb,x_l1],axis=1)
                    else:
                        X_comb=x_l1
                    tot_nodes+=lab
                A=A_l1
                x=x_l1
                in_feat=x.get_shape().as_list()[2]                
                if self.clusratio!=-1:
                    clusnext=int(clusnext*self.clusratio)
                    if clusnext==0 and i <(self.num_pool-2):
                        print(clusnext,i)
                        raise Exception('Either INCREASE pooling ratio or REDUCE number of layers as #nodes becoming zero for next layer')
                    
        X=tf.convert_to_tensor(X_comb, dtype=tf.float32)
        X_orig_comb=X  
        lis=self.shuf_list
        lis1=self.shuf_list1
        X3=np.array(X)
        lab=[]
        for ij1 in range(tot_nodes):
            lab.append([1.])
        X_hat=tf.gather(X,lis)
        X_hat=tf.concat([X_hat[i,:,:] for i in range(self.ran)],axis=1)
        X_reduce1=tf.gather(self.input_features1,lis1)
        X_reduce2=tf.concat([X_reduce1[i,:,:] for i in range(self.ran1)],axis=1)
        X_reduce=tf.concat([self.input_features1,X_reduce2],axis=1)
        for r in range(self.ran):
            for ij2 in range(tot_nodes):
                lab.append([0.])
        X_orig_comb=tf.concat([X_orig_comb,X_hat],axis=1)
        lab2=[]
        lab2.append(lab)  
        lab3=[]
        lab3.append([1.])
        for r in range(self.ran1):
            lab3.append([0.])
        lab4=[]
        lab4.append(lab3)
        lable1 = tf.tile(lab2, [tf.shape(A)[0],1,1])       
        lable2 = tf.tile(lab4, [tf.shape(A)[0],1,1]) 
        s1=s
        out_orig=self.bilinear(s,X_orig_comb)
        out_local=self.bilinear1(s,X_reduce)
        inp=self.Calc_Loss(lable2,out_local,lable1,out_orig)

    def runn(self,sess,feed1,v):
        feed={self.shuf_list:feed1['listt'],self.shuf_list1:feed1['listt1'],self.train :feed1['train1'],self.drop_rate:feed1['keep'],self.input_features:feed1['input_features1'],self.input_features1:feed1['s'],self.input_adj:feed1['input_adj1'],self.input_labels:feed1['input_labels1']}
        if v=="train":            
            run_vars = [self.train_op_np]
            c = sess.run(run_vars, feed_dict = feed)
            run_vars=[tf.trainable_variables(),tf.get_default_graph().get_tensor_by_name("Sigmoid:0"),self.loss_val,self.loss_val,self.embeddings]
            t,em1,s1,s2,em= sess.run(run_vars, feed_dict = feed)
            return s1,s2,em,em1
        if v=="test":            
            run_vars=[self.embeddings]
            em1= sess.run(run_vars, feed_dict = feed)
            return em1
                
    def optimizer(self,reuse=False):
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        with tf.name_scope('opt'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate , global_step, 100000, 0.96, staircase=True)
            self.train_op_np = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val+self.loss_val1)
  
    def _add_placeholders(self):
        self.input_features = tf.placeholder(tf.float32, shape = [None, self.num_nodes,self.feat_dim], name = "input_features")
        self.input_adj = tf.placeholder(tf.float32, shape = [None,self.num_nodes, self.num_nodes], name = "input_adj")
        self.input_features1 = tf.placeholder(tf.float32, shape = [None,1,self.feat_dim1], name = "input_s")
        self.input_labels = tf.placeholder(tf.float32, shape = [None, self.nclasses], name = "input_labels")           
        self.drop_rate = tf.placeholder(tf.float32)
        self.train = tf.placeholder(tf.bool)
        self.shuf_list=tf.placeholder(tf.int32, shape = [self.ran,None], name = "shuf_list")
        self.shuf_list1=tf.placeholder(tf.int32, shape = [self.ran1,None], name = "shuf_list")           
