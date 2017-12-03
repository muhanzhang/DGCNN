# for processing the ptc, proteins datasets into .mat format 
# ptc, proteins come from the authors of the following paper:
# Deep Graph Kernels, Pinar Yanardag, SVN Vishwanathan, KDD 2015 
# You may download their other datasets to data/raw_data/datasets/
# and process them using this script

import numpy as np
import multiprocessing as mp
import sys, copy, time, math, pickle
import scipy.io
import random
import pdb
import os


def load_data(ds_name):
    f = open(DATA_DIR + "/%s.graph"%ds_name, "r")
    data = pickle.load(f)
    graph_data = data["graph"]
    labels = data["labels"]
    labels  = np.array(labels, dtype = np.float)
    return graph_data, labels

def load_graph(nodes):
    max_size = []
    for i in nodes.keys():
        max_size.append(i)
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            max_size.append(neighbor)
    size = max(max_size)+1
    am = np.zeros((size, size))
    x = np.zeros((size, 1))
    for nidx in nodes:
        for neighbor in nodes[nidx]["neighbors"]:
            am[nidx][neighbor] = 1
            try:
                x[nidx] = nodes[nidx]['label'][0]
            except:
                x = None
    return am, x


if __name__ == "__main__":
    # location to save the results
    OUTPUT_DIR = "../data/"
    # location of the datasets
    DATA_DIR = "../data/raw_data/datasets/"
    dataNames = ['proteins', 'ptc']
    #dataNames = ['collab', 'imdb_action_romance', 'imdb_comedy_romance_scifi','reddit_iama_askreddit_atheism_trollx', 'reddit_multi_5K', 'reddit_subreddit_10K']
    G = []
    for dataname in dataNames:
        print('processing ' + dataname)
        if not os.path.isdir('../data/' + dataname):
            os.mkdir('../data/' + dataname)
        g, l = load_data(dataname)
        nlabel = None
        for gidx in range(len(g)):
            am, x = load_graph(g[gidx])
            if dataname == 'proteins' or dataname == 'ptc':
                if dataname == 'proteins':
                    x = x + 1
                if nlabel == None:
                    nlabel = x[:, 0]
                else:
                    nlabel = np.concatenate((nlabel, x[:, 0]))
                    nlabel = np.unique(nlabel)
                if dataname == 'proteins':
                    label = l[0][gidx]
                    scipy.io.savemat(OUTPUT_DIR + dataname + "/%d.mat"%(gidx+1), mdict={'A': am, 'x': x, 'y': label})
                elif dataname == 'ptc':
                    label = l[gidx]
                    if label == 1:
                        label = 2
                    elif label == -1:
                        label = 1
                    scipy.io.savemat(OUTPUT_DIR + dataname + "/%d.mat"%(gidx+1), mdict={'A': am, 'x': x, 'y': label})
            else:  # unlabeled graphs, x = ''
                x = np.ones((am.shape[0], 1))  # use 1 label instead
                try:
                    label = l[gidx]
                except:
                    label = l[0][gidx]
                if np.min(l) == -1:
                    if label == -1:
                        label = 1
                    elif label == 1:
                        label = 2
                #print(label)
                scipy.io.savemat(OUTPUT_DIR + dataname + "/%d.mat"%(gidx+1), mdict={'A': am, 'x': x, 'y': label})
        print(nlabel)

