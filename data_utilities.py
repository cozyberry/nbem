import os,sys,csv,string
import numpy as np
import random
import string
from time import localtime, strftime, time
from sklearn import metrics
from scipy import special
from pprint import pprint
def multibetaln(x):
    gammalnx=special.gammaln(x)
    sumx=np.sum(x)
    gammalnsumx=special.gammaln(sumx)
    return np.sum(gammalnx)-gammalnsumx

"""
if _labeled is True, the last column of data should be label information
data is in the form of raw data, e.g. high,2,cheap,vgood
Transform multi-value xdata to binary representation of xdata_ml.
For example, for a feature with 4 possible value: bad,good,vgood, the binary representation of a value 'bad' or 'good' or 'vgood' would be [1,0,0],[0,1,0] or [0,0,1] respectively
"""
def transform_data(data,_labeled=True):
    nfeat = np.size(data,1)
    if _labeled:
        nfeat -= 1 
    if nfeat < 1:
        raise ValueError("The feature number is invalid! It should be at least at One")

    keys = [[]]*np.size(data,1)
    numdata = -1*np.ones_like(data);

    # convert string objects to integer values for modeling:
    for k in range(np.size(data,1)):
     keys[k],garbage,numdata[:,k] = np.unique(data[:,k],True,True)

    return numdata,keys

def countData(filename):
    if (not os.path.exists(filename)) or (not os.path.isfile(filename)):
        raise ValueError("Unfound input file: %s"%filename)

    datareader = csv.reader(open(filename,'r'))
    ct = 0;
    for row in datareader:
        if ct == 0:
            ncol = len(row)
        ct = ct+1
        

    return ct,ncol

def readData(filename,ct,ncols,total=0,_random=False):
    if (not os.path.exists(filename)) or (not os.path.isfile(filename)):
        raise ValueError("Unfound input file: %s"%filename)
    ct = int(ct)
    if ct < 1:
        raise ValueError("Invalid number of rows to read"%ct)

    datareader = csv.reader(open(filename,'r'))
    if not _random:
        data = np.array(-1*np.ones((ct,ncols),float),object);
        k=0;
        for row in datareader:
            data[k,:] = np.array(row)
            k = k+1;
            if k == ct:
                break
        return data
    else:
        if total < ct:
            raise ValueError('Not enough rows! %s has %d rows in total, requiring %d rows instead.'%(filename,total,ct))
        data = np.array(-1*np.ones((total,ncols),float),object);
        k=0;
        for row in datareader:
            data[k,:] = np.array(row)
            k = k+1;
        allIDX = np.arange(total);
        random.shuffle(allIDX); # randomly shuffles allIDX order for creating 'holdout' sample
        return data[allIDX[0:ct]]

       
        

"""
This is the "example" module.

The example module supplies one function, random_gen().  For example,

"""

def random_gen(filenames,ratios=None,_random=True,):
    """
    >>> import sys
    >>> random_gen(['data/car/car.unacc.data','data/car/car.acc.data','data/car/car.good.data','data/car/car.vgood.data'])

    1:1:1:1
    """
    nfile = len(filenames)
    if nfile < 1:
        raise ValueError("Invalid input file list!: %s"%filenames)
    elif nfile > 1:
        ct = np.zeros(nfile,int)
        ncol = np.zeros(nfile,int)

        for i in range(0,nfile):
            item = filenames[i]
            if (not os.path.exists(item)) or (not os.path.isfile(item)):
                raise ValueError("Unfound input file: %s"%item)
            ct[i],ncol[i] = countData(item)
            if i >0 and ncol[i]!=ncol[i-1]:
                raise ValueError('Inconsistent Files! %s has %d columns while %s has % columns'%(filenames[i],ncol[i],filenames[i-1],ncol[i-1]))
        ct = np.array(ct,int)
        maxct = np.amax(ct)
        minct = np.amin(ct)
        sumct = np.sum(ct)
        if ratios == None:
            print 'current dataset info:'
            print ct
            line="Please input data ratio for this dataset,seperated by :. e.g. 1"
            for i in range(1,nfile):
                line+=':1'
            print line
            ratios = raw_input()
        ratios = string.split(string.strip(ratios),':')
        if len(ratios) != nfile:
            print "Unconsistent ratio for %d files. U have just input %d ratios"%(nfile,len(ratios))
        for i in range(0,len(ratios)):
            ratios[i]=float(ratios[i])
        aratios = np.array(ratios,float)
        min_k= np.argmin(aratios)
        aratios = aratios/aratios[min_k]
        ct_gen = np.divide(ct,aratios)
        min_k= np.argmin(ct_gen)
        ct_gen = np.multiply(aratios,ct_gen[min_k])
        ct_gen = np.array(ct_gen,int)
        for k in range(0,nfile):
            if k==0:
                data = readData(filenames[k],ct_gen[k],ncol[0],ct[k],_random)
            else:
                data = np.vstack((data,readData(filenames[k],ct_gen[k],ncol[0],ct[k],_random)))
        return data 
    else:
        for filename in filenames:
            ct,ncol=countData(filename)
            return readData(filename,ct,ncol)

def printStats(dist,ykeys,out=None):

    if out==None:
        out=sys.stdout

    dist = np.atleast_2d(dist)
    curnumc = np.size(dist,0)
    numc = np.size(dist,1)


    print >>out,"distribution:"
    title =""

    for i in range(0,curnumc):
        title+=',class %d'%i
    print >>out,title

    for i in range(0,numc):
        line=ykeys[i]
        for j in range(0,curnumc):
            line+=",%d"%dist[j,i]
        print >>out,line

"""
xdata is the xdata to be test in original form
ypredict is the prediction cluster by model
ydata is original label information
"""
def test_clustering(xdata,ypredict,ydata,timestamp=True,featnames=None,OUTPUTDIR=None,model=None):
    if OUTPUTDIR == None:
        OUTPUTDIR = "./"

    nfeats = np.size(xdata,1)
    if featnames != None:
        if (nfeats != len(featnames)):
            raise ValueError('The input feature info is inconstent with data of %d features'%nfeats)
    else:
        featnames = []
        for i in range(0,nfeats):
            featnames.append('feature_%d'%i)

    u,ypredict = np.unique(ypredict,return_inverse=True)
    curnumc = len(u)
    
    ykeys,ydata= np.unique(ydata,return_inverse=True)
    numrows = np.size(xdata,0)
    numc = len(ykeys)


    dist = np.zeros((curnumc,numc))
    for i in range(0,curnumc):
        a=(ypredict==i)
        for j in range(0,numc):
            oj = (ydata == j)
            dist[i][j]=np.sum(np.multiply(a,oj))

    printStats(dist,ykeys)

    outputDate=strftime("%m%d%H%M%S",localtime())
    prefix='test_clustering'
    if model != None:
        prefix='test_clustering_'+model
    if timestamp:
        outname_hu="%s_hu_%s.csv"%(prefix,outputDate)
    else:
        outname_hu="%s_hu.csv"%prefix

    out_hu=open(os.path.join(OUTPUTDIR,outname_hu),'w')

    title_hu = ""
    for attr in featnames:
        title_hu +="%s,"%attr
    title_hu+='original_class,predicted_class'

    print >> out_hu,title_hu

    for i in range(0,numrows):
        onerow_hu=""
        for j in range(0,nfeats):
            item = xdata[i][j]
            onerow_hu+="%s,"%str(item)

        onerow_hu+="%s,%d"%(str(ykeys[ydata[i]]),ypredict[i])
        print >> out_hu,onerow_hu 

    print >>out_hu,""
    ari = metrics.adjusted_rand_score(ydata,ypredict)
    ami = metrics.adjusted_mutual_info_score(ydata,ypredict)
    nmi = metrics.normalized_mutual_info_score(ydata,ypredict)
    print >>out_hu,"adjusted rand index: %f"%ari
    print "adjusted rand index: %f"%ari
    print >>out_hu,"adjusted mutual info score: %f"%ami
    print "adjusted mutual info score: %f"%ami
    print >>out_hu,"normalized mutual info score: %f"%nmi
    print "normalized mutual info score: %f"%nmi
    printStats(dist,ykeys,out_hu)
    out_hu.close()

def test():
    import doctest
    doctest.testmod()

if __name__=='__main__':
    random_gen(['data/car/car.unacc.data','data/car/car.acc.data','data/car/car.good.data','data/car/car.vgood.data'])
    #random_gen(sys.args[1:])
    #test()


