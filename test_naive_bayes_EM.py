#! /usr/bin/python

import naive_bayes_EM 
import os,sys,getopt,random,csv
import numpy as np
from data_utilities import countData,readData,random_gen


DATAPATH="/home/wei/data_processing/data/car/car.data"
DATAPATHS=['data/car/car.unacc.data','data/car/car.acc.data','data/car/car.good.data','data/car/car.vgood.data']
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
    print "%s [-c type_of_likelihood] [-n nonstochastic_iteration_times] [-s stochastic_iteration_times] [-v] [-o outputdir] [-d] [-k initial_clustering_number] [-i initial_method] [-a] [-u] [filenames]"%sys.argv[0]
    print "     [-c type_of_likelihood]: 0 for normal likelihood;1 for classification likelihood;2 for naive bayesian network. 0 By default"
    print "     [-n iteration_times]: set nonstochastic iteration times for EM method. Default is 20"
    print "     [-s stochastic_iteration_times]: set stochastic iteration times for EM method. Default is 1"
    print "     [-v]: set verbose mode. Print other detail infomation"
    print "     [-o outputdir]: specify outputdir. using %s by default"%OUTPUTDIR
    print "     [-d]: output file name with time stamp, only valid together with -o option"
    print "     [-k initial_clustering_number]: set an initial clustering number for EMNB or ECMNB."
    print """     [-i initial_method]:            set initial methods of label for EM method. 
                                     initial_method: an integer specifying the initial method 
                                     0: uniform initialization
                                     1: k points methods. Refer to PHAM2009 'Unsupervised training of bayesian networks for data clustering'"""
    print "    [-a]: set default attributes information"
    print "    [-b]: add bayes smooth operation at last"
    print "    [--alpha value_of_alpha]: specify value of alpha for prior dirichelet distribution"
    print "    [-u]: has no label info"


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hc:n:s:k:i:bvo:dpu",["help","alpha="])
    except getopt.GetoptError:
        print 'option parsing error!'
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
    global DATAPATHS
    initMethod = 0
    _PARTITION = False
    _attr = False
    _bayes= False
    numc = 4
    alpha = 2.0
    _labeled = True 
    for opt,arg in opts:
        if opt in ("-h","--help"):           
            usage()
            print arg
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
            outputdir = arg
            if (not os.path.exists(outputdir)) or (not os.path.isdir(outputdir)):
                print "Output directory: %s does not exist!"%outputdir
                usage()
                sys.exit(1)

        elif opt in ("-d"):
            _DATE= True
        elif opt in ("-p"):
            _PARTITION= True
        elif opt in ("-k"):
            numc = int(arg)
        elif opt in ("-i"):
            initMethod = int(arg)
        elif opt in ("-a"):
            _attr=True
        elif opt in ("-b"):
            _bayes=True
        elif opt in ("-u"):
            _labeled=False
        elif opt in ("--alpha"):
            print arg
            alpha=float(arg)


    if LTYPE == 0:
        random.seed()
        if len(args) > 0:
            DATAPATHS = args
        rdata = random_gen(DATAPATHS,_random=True)
        out_rdata = open('rdata.csv','w')
        writer = csv.writer(out_rdata)
        writer.writerows(rdata)
        out_rdata.close()
        nbem=naive_bayes_EM.CategoricalNBEM(alpha=alpha)
        nbem.setVerbose(_VERBOSE)
        if _OUTPUT:
            nbem.setOutput(outputdir)
        else:
            nbem.setOutput(OUTPUTDIR)

        if _labeled:
            if _attr:
                xdata_ml,ydata=nbem.fit_transformRaw(rdata,True,ATTRIBUTES)
            else:
                xdata_ml,ydata=nbem.fit_transformRaw(rdata,True)
        else:
            if _attr:
                xdata_ml=nbem.fit_transformRaw(rdata,False,ATTRIBUTES)
            else:
                xdata_ml=nbem.fit_transformRaw(rdata,False)
            
        nbem.build(numc,xdata_ml,ITERSN,ITERCN,initMethod,_DATE,_bayes=_bayes)
        if _labeled:
            nbem.testModel(xdata_ml,ydata,_DATE)
        else:
            nbem.testModel(xdata_ml,timestamp=_DATE)
            
        #out_proba = open('predict_prob','w')
        #res = nbem.predict_proba(xdata_ml)
        #for item in res:
            #print >>out_proba,item
        #out_proba.close()
    else:
        raise ValueError("Oh I just know NBEM. The corresponding LTYPE is 0")
            
if __name__=='__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main("")
