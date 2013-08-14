#! /usr/bin/python
import numpy as np
import os,sys,csv
from scipy import special
from scipy.sparse import issparse

class bnModel():
    def __init__(self,data,alpha=2.0):
        self.alpha=alpha

        keys = [[]]*np.size(data,1)
        numdata = -1*np.ones_like(data);
        self.nfeat = np.size(data,1)-1

        self.nfeatures=[0]

        for k in range(np.size(data,1)):
            keys[k],garbage,numdata[:,k] = np.unique(data[:,k],True,True)
            self.nfeatures.append(self.nfeatures[-1]+len(keys[k]))

        self.n_cluster=len(keys[-1])
        self.nfeatures.pop()
        
        self.nclass=np.ones(len(keys[-1]))
        for i in range(0,len(keys[-1])):
            self.nclass[i] = np.sum(numdata[:,-1] == i)
        print 'nclass: ',self.nclass
        
        self.nclass_feature=np.ones((len(keys[-1]),self.nfeatures[-1]))
        for i in range(0,len(keys[-1])):
                for j in range(0,self.nfeat):
                    ci=(numdata[:,-1] == i) 
                    for k in range(0,len(keys[j])):
                        fj=(numdata[:,j]==k)
                        self.nclass_feature[i][self.nfeatures[j]+k]=np.sum(np.multiply(ci,fj))
        print 'nclass_feature: ',self.nclass_feature
        print 'nfeatures: ',self.nfeatures
        print 'n_cluster: ',self.n_cluster

        

    def complete_model_ML(self):
        gamma_nclass=special.gammaln(self.nclass+self.alpha)
        gamma_nclass_feature=special.gammaln(self.nclass_feature+self.alpha)
        gamma_sum_nclass=special.gammaln(np.sum(self.nclass+self.alpha))
        nFeature=len(self.nfeatures)-1
        for i in range(0,nFeature):
            cur_gamma_sum_nclass_feature=special.gammaln(np.sum(self.nclass_feature[:,self.nfeatures[i]:self.nfeatures[i+1]]+self.alpha,axis=1))
            if i==0:
                gamma_sum_nclass_feature=cur_gamma_sum_nclass_feature
            else:
                gamma_sum_nclass_feature=np.hstack((gamma_sum_nclass_feature,cur_gamma_sum_nclass_feature))

        #pprint.pprint(locals())
        nparams=self.n_cluster+self.n_cluster*(self.nfeatures[-1])
        gamma_sum_alpha=special.gammaln(self.n_cluster*self.alpha)
        for i in range(0,nFeature):
            gamma_sum_alpha+=self.n_cluster*special.gammaln(self.alpha*(self.nfeatures[i+1]-self.nfeatures[i]))

        return np.sum(gamma_nclass_feature)+np.sum(gamma_nclass)-np.sum(gamma_sum_nclass_feature)-gamma_sum_nclass+gamma_sum_alpha-nparams*special.gammaln(self.alpha)

if __name__=='__main__':
    filename=sys.argv[1]
    csvreader = csv.reader(open(filename,'r'))
    ct = 0
    for row in csvreader:
        if ct==0:
            ncol=len(row)
        ct+=1

    print 'ct: ',ct
    print 'ncol: ',ncol
    data=np.ndarray(shape=(ct,ncol),dtype=object)
    csvreader = csv.reader(open(filename,'r'))
    k=0
    for row in csvreader:
        data[k,:]=np.array(row,dtype='str')
        k+=1

    bnm=bnModel(data)
    print bnm.complete_model_ML()
