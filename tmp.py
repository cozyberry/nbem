#! /usr/bin/python
import sys,os
import numpy as np
import pprint
from scipy.special import gammaln

if __name__=='__main__':
    N=np.zeros((3,2,2))
    #alpha=2*np.ones((3,2,2))
    alpha=2.0
    a=int(sys.argv[1])
    b=int(sys.argv[2])
    r=[2,2,2]
    q=[1,2,2]
    for i in range(2):
        N[0,0,i]=a

    for i in range(1,3):
        for j in range(q[i]):
            for k in range(r[i]):
                N[i,j,k]=b
    res=0.0
    for i in range(3):
        for j in range(q[i]):
            for k in range(r[i]):
                res+=gammaln(alpha+N[i,j,k])-gammaln(alpha)
                #print i,j,k,N[i,j,k]

    for i in range(3):
        for j in range(q[i]):
            res+=gammaln(2*alpha)-gammaln(2*alpha+np.sum(N[i,j,:]))
    print res
