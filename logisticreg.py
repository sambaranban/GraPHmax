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
class logistic(object):
    def __init__(self,placeholders):
        
        self.num_nodes=placeholders['num_nodes']
        self.feat_dim=placeholders['feat_dim']
        self.output_dim=placeholders['emb_dim']
        
        self.nclasses=placeholders['nclasses']
        self.learning_rate=placeholders['learning_rate']
        self.batch_size=placeholders['batch_size']
            
    def runn(self,sess,feed1,v):
        feed={self.train :feed1['train1'],self.drop_rate:feed1['keep'],self.features:feed1['input_features1'],self.input_labels:feed1['input_labels1']}
        if v=="train":            
            run_vars = [self.train_op_np]
            c = sess.run(run_vars, feed_dict = feed)
            run_vars=[self.log_loss,self.log_acc]
            s1,a1= sess.run(run_vars, feed_dict = feed)
            return s1,a1
        if v=="test" or v=="val":            
            run_vars=[self.log_loss,self.log_acc]
            s1,a1= sess.run(run_vars, feed_dict = feed)
            return s1,a1
        
        
    def optimizer(self,reuse=False):
        global_step = tf.Variable(0, name = "global_step", trainable = False)
        with tf.name_scope('opt'):
            self.learning_rate = tf.train.exponential_decay(self.learning_rate , global_step, 100000, 0.96, staircase=True)
            self.train_op_np = tf.train.AdamOptimizer(self.learning_rate).minimize(self.log_loss)#_val+self.loss_val1)
  
    def _add_placeholders(self):
        self.features = tf.placeholder(tf.float32, shape = [None,self.feat_dim], name = "input_features")
        self.input_labels = tf.placeholder(tf.float32, shape = [None, self.nclasses], name = "input_labels")           
        self.drop_rate = tf.placeholder(tf.float32)
        self.train = tf.placeholder(tf.bool)
                
    def logistic_loss(self,logits,labels):
        # Weight decay loss
#         print("log",logits)
#         print("lab",labels)
        self.log_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        self.optimizer()
            
    def logistic_accuracy(self,output):
        y_pred =tf.nn.softmax(output)
        y_pred_cls = tf.argmax(y_pred, dimension=1,name='pp')         
        l=tf.argmax(self.input_labels, dimension=1,name="pp1")
        correct_prediction = tf.equal(y_pred_cls, l)       
        self.log_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="mean")       
       

    def logistic_build(self):
        self._add_placeholders()
        self.input_dim=self.features.get_shape().as_list()[-1]
        weights = glrt_init([self.input_dim, self.output_dim],name=None)
        x = tf.nn.dropout(self.features, 1-self.drop_rate)
        self.log_outputs=tf.tensordot(x,weights,axes=[[-1],[0]])
#         self.log_outputs=tf.math.l2_normalize(self.log_outputs,-1)
        self.logistic_loss(self.log_outputs,self.input_labels)
        self.logistic_accuracy(self.log_outputs)


# In[17]:
