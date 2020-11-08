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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
from hierarchical import *
from periphery import *
from logisticreg import *


seed(123)
random.seed(123)
np.random.seed(123)
# In[2]:

def glrt_init(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
def one(i,n):
    a = np.zeros(n, 'uint8')  
    a[i] = 1
    return a
       

################################ THIS FUNCTION (read_graphfile) IS ADAPTED FROM RexYing/HybridPool ############################################

def read_graphfile(dataname):
    max_nodes=None
    #read datasets
    prefix='dataset_graph/'+dataname+'/'+dataname
    data_list=[]
    data={}
    filename_graph_indic = prefix + '_graph_indicator.txt'
    # index of graphs that a given node belongs to
    graph_indic={}
    with open(filename_graph_indic) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    filename_nodes=prefix + '_node_labels.txt'
    node_labels=[]
    try:
        with open(filename_nodes) as f:
            for line in f:
                line=line.strip("\n")
                node_labels+=[int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
 
    filename_node_attrs=prefix + '_node_attributes.txt'
    node_attrs=[]
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
       
    label_has_zero = False
    filename_graphs=prefix + '_graph_labels.txt'
    graph_labels=[]

    # assume that all graph labels appear in the dataset 
    #(set of labels don't have to be consecutive)
    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)
    
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
    
    filename_adj=prefix + '_A.txt'
    adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
     
    for i in range(1,1+len(adj_list)):
        # indexed from 1 here
        G=nx.from_edgelist(adj_list[i])       
        
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
      
        # add features and labels
        G.graph['label'] = graph_labels[i-1]
        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u-1]
                node_label_one_hot[node_label] = 1
                G.node[u]['label'] = np.array(node_label_one_hot)
            if len(node_attrs) > 0:
                G.node[u]['feat'] = node_attrs[u-1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]
        mapping={}
        it=0
        if float(nx.__version__)<2.0:
            for n in G.nodes():
                mapping[n]=it
                it+=1
        else:
            for n in G.nodes:
                mapping[n]=it
                it+=1
            
       
        graphs.append(nx.relabel_nodes(G, mapping))

    max_num_nodes = max([G.number_of_nodes() for G in graphs])
    if len(node_attrs)>0:
        feat_dim = graphs[0].node[0]['feat'].shape[0]
    lab1=[]
    feat_dim1 = graphs[0].node[0]['label'].shape[0]

    for G in graphs:        
        adj = np.array(nx.to_numpy_matrix(G))

        num_nodes = adj.shape[0]
        adj_padded = np.zeros((max_num_nodes,max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        label1=G.graph['label']
        
        if len(node_attrs)>0:
            f = np.zeros((max_num_nodes,feat_dim), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['feat']
            
            f1=np.identity(max_num_nodes)
            f = np.concatenate((f, f1), axis=1)
            rowsum = np.array(f.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = np.diag(r_inv)
            f = r_mat_inv.dot(f)

        else:

            max_deg = 10
            f = np.zeros((max_num_nodes,feat_dim1), dtype=float)
            for i,u in enumerate(G.nodes()):
                f[i,:] = G.node[u]['label']
            degs = np.sum(np.array(adj), 1).astype(int)

            degs[degs>max_deg] = max_deg
            feat = np.zeros((len(degs), max_deg + 1))
            feat[np.arange(len(degs)), degs] = 1
            feat = np.pad(feat, ((0, max_num_nodes - G.number_of_nodes()), (0, 0)),
                    'constant', constant_values=0)

            f = np.concatenate((feat, f), axis=1)
            f1=np.identity(max_num_nodes)
            rowsum = np.array(f.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = np.diag(r_inv)
            f = r_mat_inv.dot(f)
        lab1.append(label1) 
        label1=one(label1,len(label_vals))
        data={}
        data['feat']=f
        data['adj']=adj_padded
        data['label']=label1
        data_list.append(data)
    return data_list, len(label_vals),max_num_nodes,lab1
# In[3]:


def periphery_1(adj,labels,feat,arguments,nclasses,max_num_nodes):
	####MAIN FUNCTION####
	placeholders={}
	final={}
	train_adj=adj
	train_label=labels
	train_feat=feat

	placeholders={'gcn_dim':[128,128,64,64],'gcn_encoder_layers':2,'feat_dim':feat[0].shape[-1],'learning_rate':0.001,'num_nodes':np.array(adj).shape[-1],'emb_dim':feat[0].shape[-1],'nclasses':np.array(labels).shape[-1]}
	              
	a1=0
	pat=20 
	tf.reset_default_graph()
	tf.set_random_seed(123)
	D=gin_gae(placeholders)

	# idx=np.arrange(len(adj))
	######Change batch_size according to the dataset######
	batch_size=len(train_adj)
	num_batches=int(len(train_adj)/batch_size)
	D.gae_architechture()
	# #####to set GPU ##########
	# # config = tf.ConfigProto(device_count = {'GPU': 4})
	# # config.gpu_options.allow_growth=False
	sess = tf.Session()#config=config)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	asqmx=0.0
	step = 0
	embeddings=[]
	cost_val=[]
	for epoch in range(2000):
	    embeddings=[]
	    trainavrloss = 0
	    trainavracc = 0
	    i10=0
	    tr_step = 0
	    for j in range(num_batches):            
	        feed = {}
	        adj1=np.reshape(train_adj[j*batch_size:j*batch_size+batch_size],[-1])
	        pos_weight = float(adj1.shape[0] - adj1.sum()) / adj1.sum()
	        norm = adj1.shape[0]/ float((adj1.shape[0]- adj1.sum()) * 2)
	        feed['input_features1'] = train_feat[j*batch_size:j*batch_size+batch_size]
	        feed['input_adj1'] = train_adj[j*batch_size:j*batch_size+batch_size]
	        feed['input_labels1']=train_label[j*batch_size:j*batch_size+batch_size]
	        feed['keep']=0.1
	        feed['pos']=pos_weight
	        feed['norm']=norm
	        feed['train1']=True
	        summ1,embd=D.runn(sess,feed,"train")
	        trainavrloss += summ1
	        tr_step += 1  
	               
	    cost_val.append(trainavrloss/tr_step)
	    print('epoch:',epoch,'   Training: loss::', trainavrloss/tr_step)
	    if epoch > 30 and cost_val[-1] > np.mean(cost_val[-(30+1):-1]):
	        print("Early stopping...")
	        break
	        
	feed = {}
	adj1=np.reshape(train_adj[j*batch_size:j*batch_size+batch_size],[-1])
	pos_weight = float(adj1.shape[0] - adj1.sum()) / adj1.sum()
	norm = adj1.shape[0]/ float((adj1.shape[0]- adj1.sum()) * 2)
	feed['input_features1'] = train_feat
	feed['input_adj1'] = train_adj
	feed['input_labels1']=train_label
	feed['keep']=0
	feed['pos']=pos_weight
	feed['norm']=norm
	feed['train1']=False
	s,embeddings=D.runn(sess,feed,"test")

	print("Training Completed")

	sess.close()

	return embeddings


def hierarchical_2(adj,labels,feat,EF,arguments,nclasses,max_num_node):

		
	train_adj=adj
	train_label=labels
	train_feat=feat
	tf.reset_default_graph()
	tf.set_random_seed(123)
	np.random.seed(123)
	batch_size=len(train_adj)
	embeddings_last=EF
	s=train_feat-embeddings_last
	s=np.array(s)
	print(s.shape[-1])
	s=np.mean(s,axis=1,keepdims=True)

	ran=4 
	ran1=4
	num_pool=2 
	clusrstio=0.5
	placeholders={'feat_dim1':s.shape[-1],'batch_size':batch_size,'ran':ran,'gcn_dim':128,'emb_dim':128,'feat_dim':feat[0].shape[1],'learning_rate':0.001,'num_nodes':max_num_node,'num_pool':num_pool,'outp_dim':128,'clusrstio':clusrstio,'nclasses':nclasses}

	D=hierarchical_repr(placeholders)

	num_batches=int(len(train_adj)/batch_size)
	D.hierarchical_repr_arch()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	asqmx=0.0
	step = 0

	node_embeddings=[]
	rand1=[ii for ii in range(batch_size)]

	cost_val=[]
	for epoch in range(10000):
	    randd1=[]
	    for kl in range(ran):
	        idx = np.random.permutation(np.arange(0, batch_size)).tolist()
	        randd1.append(idx)

	    randd2=[]
	    for kl in range(ran1):
	        idx = np.random.permutation(np.arange(0, batch_size)).tolist()
	        randd2.append(idx)
	    trainavrloss = 0
	    trainavracc = 0
	    i10=0
	    batch_size=len(train_adj)
	    tr_step = 0
	    tr_size = len(train_adj)
	    num_batches=int(len(train_adj)/batch_size)
	    for j in range(num_batches):   
	        feed = {}
	        feed['input_features1'] = train_feat[j*batch_size:j*batch_size+batch_size]
	        feed['input_adj1'] = train_adj[j*batch_size:j*batch_size+batch_size]
	        feed['s']=s[j*batch_size:j*batch_size+batch_size]
	        feed['input_labels1']=train_label[j*batch_size:j*batch_size+batch_size]
	        feed['keep']=0.2
	        feed['train1']=True
	        feed['listt']=randd1
	        feed['listt1']=randd2
	        s1,s2,embd,em1=D.runn(sess,feed,"train")
	        trainavrloss += s1+s2
	        tr_step += 1  

	    
	    print('epoch',epoch,'Training: loss::', trainavrloss/tr_step)#, 'accuracy::',trainavracc/tr_step)           
	    cost_val.append(trainavrloss/tr_step)
	    if epoch > 1550 and cost_val[-1] > np.mean(cost_val[-(1550+1):-1]):
	        print("Early stopping...")
	        break
	print("Training Completed")

	feed = {}
	feed['input_features1'] = train_feat
	feed['input_adj1'] = train_adj
	feed['s']=s
	feed['input_labels1']=train_label
	feed['keep']=0.
	feed['train1']=False
	feed['listt']=randd1
	feed['listt1']=randd2

	embd=D.runn(sess,feed,"test")
	tr_step += 1  
	node_embeddings=embd
	sess.close()

	feats=np.array(node_embeddings[-1])
	print(feats.shape)
	feats=np.reshape(feats,[train_feat.shape[0],feats.shape[-1]])
	print(feats.shape)

	l1=[np.where(r==1)[0][0] for r in train_label]
	e=feats
	
	return feats


def logistic_3(adj,labels,feat,lab1,feats,arguments,nclasses,max_num_node):


	train_adj=adj
	train_label=labels
	train_feat=feat
	lab1=np.array(lab1)
	######Change batch_size according to the dataset######
	pat=10
	
	print(feats.shape)

	
	kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=0) #KFold(n_splits=10)
	ep=[[] for ir in range(10)]
	it=0
	for train_index, test_index in kf.split(train_adj,lab1):
	    train_label,test_label=labels[train_index],labels[test_index]
	    train_feat,test_feat=feats[train_index],feats[test_index]	            
	    tf.reset_default_graph()
	    batch_size=len(train_feat)
	    
	    tf.set_random_seed(123)
	    nclasses=train_label[0].shape[-1]
	    max_no_node=train_feat.shape[1]
	    placeholders={'batch_size':batch_size,'emb_dim':labels[0].shape[-1],'feat_dim':feats.shape[1],'learning_rate':0.05,'num_nodes':max_num_node,'nclasses':nclasses}
	    D=logistic(placeholders)
	    num_batches=int(len(train_feat)/batch_size)
	    D.logistic_build()

	    sess = tf.Session()
	    sess.run(tf.global_variables_initializer())
	    sess.run(tf.local_variables_initializer())
	    vlss_mn = np.inf
	    vacc_mx = 0.0
	    asqmx=0.0
	    step = 0
	    # it=0    
	    for epoch in range(5000):
	        trainavrloss = 0
	        trainavracc = 0
	        vallossavr = 0
	        valaccravr = 0
	        i10=0
	        
	        batch_size=len(train_feat)
	        
	        tr_step = 0
	        tr_size = len(train_feat)
	        for j in range(num_batches):            
	            feed = {}
	            feed['input_features1'] = train_feat[j*batch_size:j*batch_size+batch_size]
	            feed['input_labels1']=train_label[j*batch_size:j*batch_size+batch_size]
	            feed['keep']=0.3
	            feed['train1']=True
	            summ1,a1=D.runn(sess,feed,"train")
	            trainavrloss += summ1
	            trainavracc += a1
	            tr_step += 1

	        feed = {}
	        i10=0
	        batch_size=len(test_feat)

	        feed['input_features1'] = test_feat
	        feed['input_labels1']=test_label
	        feed['keep']=0
	        feed['train1']=False

	        summ,a=D.runn(sess,feed,"val")
	        ep[it].append(a*100)
	    it+=1
	
	ep1=np.mean(ep,axis=0)
	ep11=ep1.tolist()
	epi=ep11.index(max(ep11))
	print(epi,max(ep11))



def argument_parser():
    parser = argparse.ArgumentParser(description="GraPHmax for graph classification")
    parser.add_argument("-dt", "--dataset", type=str, help="name of the dataset", default="MUTAG")
    parser.add_argument("-ll", "--lrl",type=float, default=0.05, help="learning rate logistic")
    # parser.add_argument("-ss", "--sub_samp", type=int, default=12, help="number of subgraphs to be sampled")

    parser.add_argument("-lp", "--lrp",  type=float, default=0.001, help="learning rate periphery")
    parser.add_argument("-lh", "--lrh",type=float, default=0.001, help="learning rate hierarchical")
    parser.add_argument("-pr", "--pool_rt", type=float, default=0.1, help="pooling ratio")

    parser.add_argument("-pl", "--pool_lay", type=int, default=2, help="number of pooling layers")
    parser.add_argument("-ngpr", "--negpr", type=int, default=4, help="number of negative samples periphery")
    parser.add_argument("-nghr", "--neghr", type=int, default=4, help="number of negative samples hierarchical")

    parser.add_argument("-ed", "--embd_dim", type=int, default=128, help="embedding dimension")
    parser.add_argument("-dr", "--dropout", type=float, default=0, help="dropout rate")

    arguments = parser.parse_args()
    return arguments
def read_dataset(dataset):
    ################### read in graphs ########################
    datasets,n_classes,max_num_node,lab1=read_graphfile(dataset)
    datasets=np.array(datasets)
    return datasets,n_classes,max_num_node,lab1

def main():
    arguments = argument_parser()
    dataset,nclasses,max_num_nodes,lab1=read_dataset(arguments.dataset)
    #################SEPERATE EACH COMPONENT###################
    adj=[]
    labels=[]
    feat=[]
    subgraphs1=[]
    for i in range(len(dataset)):
        adj.append(dataset[i]['adj'])
        labels.append(dataset[i]['label'])
        feat.append(dataset[i]['feat'])
    print(len(adj),len(labels),len(feat))

    adj=np.array(adj)
    labels=np.array(labels)
    feat=np.array(feat)

    EF=periphery_1(adj,labels,feat,arguments,nclasses,max_num_nodes)
    representation=hierarchical_2(adj,labels,feat,EF,arguments,nclasses,max_num_nodes)
    logistic_3(adj,labels,feat,lab1,representation,arguments,nclasses,max_num_nodes)
#Main Function
if __name__ == "__main__":
    main()

