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

class gin_gae(object):
    def __init__(self,placeholders):
        self.num_nodes=placeholders['num_nodes']
        self.gcn_encoder_layers=placeholders['gcn_encoder_layers']
        self.feat_dim=placeholders['feat_dim']
        self.embd_dim=placeholders['emb_dim']
        self.gcn_dim=placeholders['gcn_dim']
        self.nclasses=placeholders['nclasses']
        self.learning_rate=placeholders['learning_rate']

    def encoder(self,A1,H,out_feat,in_feat,act,name='gcn',i=0,k=1,k1=0.3):
        weights = glrt_init([in_feat, out_feat],name=name)
        n12=A1.get_shape().as_list()
        eps=tf.Variable(tf.zeros(1))
        rowsum = tf.reduce_sum(A1,axis=2)            
        d_inv_sqrt = tf.contrib.layers.flatten(tf.rsqrt(rowsum))
        d_inv_sqrt=tf.where(tf.is_inf(d_inv_sqrt), tf.zeros_like(d_inv_sqrt), d_inv_sqrt)        
        d_inv_sqrt=tf.linalg.diag(d_inv_sqrt)       
        A1=tf.matmul(tf.matmul(d_inv_sqrt,A1),d_inv_sqrt)
        H = tf.nn.dropout(H, 1-k1)
        A1=tf.matmul(A1,H,name=name+'matmul1')
        A1=A1+(1+eps)*H 
        for i in range(2-1):
          ad=tf.keras.layers.Dense(units = out_feat)(A1)
          ab=tf.keras.layers.BatchNormalization()(ad)
          input_features = tf.nn.relu(ab)

        H_next=tf.keras.layers.Dense(units = out_feat)(A1)#(input_features)
        ab=tf.keras.layers.BatchNormalization()(H_next)
        H_next = tf.nn.relu(ab)
        return H_next
    

    def autoencoder_loss(self,A_hat,A):
        
        AHAH=tf.matmul(A_hat,tf.transpose(A_hat,[0,2,1]))
        self.outputs=AHAH
        self.outputs2=tf.nn.sigmoid(self.outputs)
        logits = tf.reshape(self.outputs, [-1])
        labels = tf.reshape(A, [-1])
        self.loss_val_cross=self.norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=self.pos_weight))
              
        self.optimizer_encoder()       
    
    def gae_architechture(self):
        self._add_placeholders()
        A=self.input_adj  
        x=self.input_features
        in_feat=x.get_shape().as_list()[2]            
        for lay in range(self.gcn_encoder_layers-1):
            x1=self.encoder(A,x,self.gcn_dim[lay],in_feat,act="relu",name='gcn',i=0,k=self.train,k1=self.drop_rate)
            x=x1
            in_feat=x.get_shape().as_list()[2]
        in_feat=x.get_shape().as_list()[2]
        x1=self.encoder(A,x,self.embd_dim,in_feat,act=None,name='gcn',i=0,k=self.train,k1=self.drop_rate)
        x=x1
        self.outputs1=x1
        self.autoencoder_loss(x,A)

    def optimizer_encoder(self,reuse=False):
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        with tf.name_scope('opt'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate , global_step, 100000, 0.96, staircase=True)
            self.train_op_np = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_val_cross)#self.loss_val)
    
    def runn(self,sess,feed1,v): 
        feed={self.pos_weight:feed1['pos'],self.norm:feed1['norm'],self.train :feed1['train1'],self.drop_rate:feed1['keep'],self.input_features:feed1['input_features1'],self.input_adj:feed1['input_adj1'],self.input_labels:feed1['input_labels1']}
        if v=="train":            
            run_vars = [self.train_op_np]
            c = sess.run(run_vars, feed_dict = feed)  
            run_vars=[self.loss_val_cross,self.outputs]
            summ,emb= sess.run(run_vars, feed_dict = feed)
            return summ,emb
        if v=="test":            
            run_vars=[self.loss_val_cross,self.outputs1]
            summ,emb= sess.run(run_vars, feed_dict = feed)
            return summ,emb
        
    def _add_placeholders(self):
        self.input_features = tf.placeholder(tf.float32, shape = [None, self.num_nodes,self.feat_dim], name = "input_features")
        self.input_adj = tf.placeholder(tf.float32, shape = [None,self.num_nodes, self.num_nodes], name = "input_adj")
        self.pos_weight = tf.placeholder(tf.float32)
        self.norm = tf.placeholder(tf.float32)
        
        self.input_labels = tf.placeholder(tf.int32, shape = [None, self.nclasses], name = "input_labels")           
        self.drop_rate = tf.placeholder(tf.float32)
        self.train = tf.placeholder(tf.bool)
        