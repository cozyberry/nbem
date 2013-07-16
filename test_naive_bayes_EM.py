#! /usr/bin/python

import naive_bayes_EM 
import os,sys,getopt,random,csv
import numpy as np

DATAPATH="/home/wei/data_processing/data/car/car.data"
ITERCN = 20
ITERSN = 1
_VERBOSE = False
_MAXLOG  = False
_OUTPUT  = False
_DATE    = False
ATTRIBUTES = ['buyPrice','maintPrice','numDoors','numPersons','lugBoot','safety']
OUTPUTDIR ='/home/wei/share/nbem/outputs/'
LTYPE = 0

def usage():
    print "%s [-c type_of_likelihood] [-n nonstochastic_iteration_times] [-s stochastic_iteration_times] [-v] [-o] [-d] [-k initial_clustering_number] [-i initial_method]"%sys.argv[0]
    print "     [-c type_of_likelihood]: 0 for normal likelihood;1 for classification likelihood;2 for naive bayesian network. 0 By default"
    print "     [-n iteration_times]: set nonstochastic iteration times for EM method. Default is 20"
    print "     [-s stochastic_iteration_times]: set stochastic iteration times for EM method. Default is 1"
    print "     [-v]: set verbose mode. Print other detail infomation"
    print "     [-l]: set objective of maximization of log likelihood; by default maximiation of score. Need to analysize further"
    print "     [-o]: output predicted class label and original label as well for further analysis"
    print "     [-d]: output file name with time stamp, only valid together with -o option"
    print "     [-p]: set partition mode."
    print "     [-k initial_clustering_number]: set an initial clustering number for EMNB or ECMNB."
    print """     [-i initial_method]:            set initial methods of label for EM method. 
                                     initial_method: an integer specifying the initial method 
                                     0: uniform initialization
                                     1: k points methods. Refer to PHAM2009 'Unsupervised training of bayesian networks for data clustering'"""

def readData(filename):
    if not os.path.exists(filename):
        print "I can't find this file: %s"%filename
        sys.exit(1)

    datareader = csv.reader(open(filename,'r'))
    ct = 0;
    for row in datareader:
        ct = ct+1

    datareader = csv.reader(open(filename,'r'))
    data = np.array(-1*np.ones((ct,7),float),object);
    k=0;

    for row in datareader:
        data[k,:] = np.array(row)
        k = k+1;

    return data


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hc:n:s:k:i:vodp",["help"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    global ITERCN
    global ITERSN
    global _VERBOSE
    global _MAXLOG
    global _OUTPUT
    global _DATE
    global LTYPE
    global OUTPUTDIR 
    global LOGDIR 
    initMethod = 0
    _PARTITION = False
    numc = 4
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            sys.exit(0)
        elif opt in ("-c"):
            LTYPE = int(arg)
            #if LTYPE != 0 and LTYPE !=1 and LTYPE!=2:
                #print "Oh I don't know this type of likelihood: %d"
        elif opt in ("-n"):
            ITERCN = int(arg)
        elif opt in ("-s"):
            ITERSN = int(arg)
        elif opt in ("-v"):
            _VERBOSE = True
        #elif opt in ("-l"):
            #_MAXLOG= True
        elif opt in ("-o"):
            _OUTPUT= True
        elif opt in ("-d"):
            _DATE= True
        elif opt in ("-p"):
            _PARTITION= True
        elif opt in ("-k"):
            numc = int(arg)
        elif opt in ("-i"):
            initMethod = int(arg)

    if LTYPE == 0:
        random.seed()
        rdata = readData(DATAPATH)
        nbem=naive_bayes_EM.MultinomialNBEM()
        nbem.setVerbose(_VERBOSE)
        if _OUTPUT:
            nbem.setOutput(OUTPUTDIR)
        xdata_ml,ydata=nbem.fit_transformRaw(rdata,True,ATTRIBUTES)
        nbem.build(numc,xdata_ml,ITERSN,ITERCN,initMethod,_DATE)
        nbem.testModel(xdata_ml,ydata,_DATE)
    else:
        raise ValueError("Oh I just know NBEM. The corresponding LTYPE is 0")
            
if __name__=='__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main("")
