#! /usr/bin/python

import naive_bayes_EM 
import os,sys,getopt,random,csv
import numpy as np
from sklearn import cluster
from data_utilities import countData,readData,random_gen,test_clustering,transform_data

DATAPATHS=['data/car/car.unacc.data','data/car/car.acc.data','data/car/car.good.data','data/car/car.vgood.data']
OUTPUTDIR='/home/wei/share/kmeans/'

def usage():
    print "%s [-h] [-o] [-k initial_clustering_number] [-d] [filenames]"%sys.argv[0]
    print "     [-h]: show help message"
    print "     [-o]: output predicted class label and original label as well for further analysis"
    print "     [-k initial_clustering_number]: set an initial clustering number for EMNB or ECMNB."
    print "     [-d]: add timestamp with output file"

def kmeans(xdata,n_cluster):
    k_means = cluster.KMeans(n_clusters=n_cluster)
    k_means.fit(xdata)
    ypredict = k_means.labels_
    return ypredict

def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hk:od",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    global DATAPATHS 
    _OUTPUT = False
    numc = 4
    _labeled = True
    _Date = False

    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-o"):
            _OUTPUT= True
        elif opt in ("-k"):
            numc = int(arg)
        elif opt in ("-d"):
            _Date = False

    if len(args) > 0:
        DATAPATHS = args
    rdata = random_gen(DATAPATHS,_random=True)
    data,keys=transform_data(rdata,_labeled)
    if _labeled:
        xkeys = keys[:-1]
        xdata = data[:,:-1]
        ykeys = keys[-1]
        ydata = data[:,-1]
    else:
        xdata = data
        xkeys = keys

    ypredict = kmeans(xdata,numc)
    test_clustering(rdata[:,:-1],ypredict,rdata[:,-1],timestamp=_Date,OUTPUTDIR=OUTPUTDIR,model='kmeans')



if __name__=='__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        print "No arguments? Really? Are u serious?"
        usage()

        
