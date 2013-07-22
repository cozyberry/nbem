#! /usr/bin/python

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import cluster
import numpy as np
from numpy import linalg as LA
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
import os,sys
import string
from time import localtime, strftime, time
import itertools
import random
import copy
from scipy import special

class BaseMultinomialNBEM(naive_bayes.MultinomialNB):
    """
    a Multinomial Naive Bayes Cluster which combines [naive bayes classifier implementation] and [EM or ECM method] to deal with missing label information. 

    The Multinomial Naive Bayes Cluster is suitable for clustering with
    discrete features (e.g., word counts for text classification). 

    The Multinomial Naive Bayes Cluster can accept data without or with label information. 
    But the label information would only be used as informative guides

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size=[n_classes,]
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    iterSN : iteration number for EM
    iterCN : retrial number for EM

    Attributes
    ----------

    n_cluster: initial maximum number of cluster

    local_prob_table: local probability table for each feature node. shape=[n_classes,n_features]

    featIndex: an internal array index for multi-value features. shape=[sum of domain length of each feature]

    nfeatures: an internal array of domain length of each multi-value feature shape=[feature number+1]

    `intercept_`, `class_log_prior_` : array, shape = [n_classes]
        Smoothed empirical log probability for each class.

    `feature_log_prob_`, `coef_` : array, shape = [n_classes, n_features]
        Empirical log probability of features
        given a class, P(x_i|y).

        (`intercept_` and `coef_` are properties
        referring to `class_log_prior_` and
        `feature_log_prob_`, respectively.)

    Examples
    --------
    To be modified
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, Y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2]))
    [3]

    Notes
    -----
    To be modified
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        naive_bayes.MultinomialNB.__init__(self,alpha,fit_prior,class_prior)
        self._verbose = False
        self.outputDir = None 
        if fit_prior == True and class_prior:
            print "The fit_prior and class_prior are both set as True. We will use the assigned class_prior and the fit_prior will be ignored"

    def setVerbose(self,verbose):
        self._verbose = verbose

    def setOutput(self,outputdir):
        if (not os.path.exists(outputdir)) or (not os.path.isdir(outputdir)):
            raise ValueError("The output directory is invalide %s"%outputdir) 
            return False 
        self.outputDir = outputdir 

        


    """
    if _labeled is True, the last column of data should be label information
    data is in the form of raw data, e.g. high,2,cheap,vgood
    Transform multi-value xdata to binary representation of xdata_ml.
    For example, for a feature with 4 possible value: bad,good,vgood, the binary representation of a value 'bad' or 'good' or 'vgood' would be [1,0,0],[0,1,0] or [0,0,1] respectively
    """
    def fit_transformRaw(self,data,_labeled=True,arrayAttr=None):
        self._labeled=_labeled

        if arrayAttr != None:
            self.featnames = np.array(arrayAttr,str)
        else:
            nfeat = np.size(data,1)
            if _labeled:
                nfeat -= 1 
            if nfeat < 1:
                raise ValueError("The feature number is invalid! It should be at least at One")
            tmp = []
            for i in range(0,nfeat):
                tmp.append('feature_%d'%i)

            self.featnames = np.array(tmp,str)  

        keys = [[]]*np.size(data,1)
        numdata = -1*np.ones_like(data);

        # convert string objects to integer values for modeling:
        for k in range(np.size(data,1)):
         keys[k],garbage,numdata[:,k] = np.unique(data[:,k],True,True)

        numrows = np.size(numdata,0); # number of instances in car data set

        if _labeled:
            xdata = numdata[:,:-1]; # x-data is all data BUT the last column which are the class labels
            ydata = numdata[:,-1]; # y-data is set to class labels in the final column, signified by -1
            self.xkeys = keys[:-1]
            self.ykeys = keys[-1]
        else:
            xdata = numdata
            self.xkeys = keys

        self.nfeatures=[0]
        lbin = LabelBinarizer();
        self.unique_feats=[]

        for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
            if k==0:
                xdata_ml = lbin.fit_transform(xdata[:,k]);
                if len(lbin.classes_) == 2:
                    xdata_ml = np.concatenate((1 - xdata_ml, xdata_ml), axis=1)
                if len(lbin.classes_) == 1:
                    xdata_ml = 1-xdata_ml
                    self.unique_feats.append(self.nfeatures[-1])
                self.featIndex = lbin.classes_
                    
                self.nfeatures.append(len(lbin.classes_))
            else:
                cur_xdata_ml=lbin.fit_transform(xdata[:,k])
                if len(lbin.classes_) == 2:
                    cur_xdata_ml = np.concatenate((1 - cur_xdata_ml, cur_xdata_ml), axis=1)
                if len(lbin.classes_) == 1:
                    self.unique_feats.append(self.nfeatures[-1])
                    cur_xdata_ml = 1-cur_xdata
                xdata_ml = np.hstack((xdata_ml,cur_xdata_ml))
                self.featIndex= np.hstack((self.featIndex,lbin.classes_))
                self.nfeatures.append(self.nfeatures[-1]+len(lbin.classes_))
    
        #print self.unique_feats
        if _labeled:
            return xdata_ml,ydata
        else:
            return xdata_ml

    def uniform_init_theta(self,xdata_ml):
        np.random.seed()
        dim = self.n_cluster+np.size(xdata_ml,1)*self.n_cluster
        array_alpha = np.ones(dim)*self.alpha
        itheta = np.random.dirichlet(array_alpha,1).transpose()
        #self.class_log_prior_ = np.ndarray(shape=(self.n_cluster,),dtype=float)
        for i in range(0,self.n_cluster):
            self.class_log_prior_=itheta[i,0]
        #self.feature_log_prob_= np.ndarray(shape=(self.n_cluster,np.size(xdata_ml,1)),dtype=float)
        for i in range(0,self.n_cluster):
            for j in range(0,np.size(xdata_ml,1)):
                self.feature_log_prob_[i,j] = itheta[self.n_cluster+np.size(xdata_ml,1)*i+j,0]


    def kmeans_init(self,xdata_ml):
        k_means = cluster.KMeans(n_clusters=self.n_cluster)
        k_means.fit(xdata_ml)
        ypredict = k_means.labels_
        return ypredict

    def uniform_init_from_data(self,xdata_ml):
        numc = self.n_cluster
        random.seed()
        numrows = np.size(xdata_ml,0)
        ytrain = -1*np.ones(numrows);
        for k in range(0,numrows):
            #randint is inclusive in both end
            ytrain[k]=random.randint(0,numc-1)
        return ytrain

    def k_points_init(self,xdata_ml):
        numrows = np.size(xdata_ml,0)
        ytrain = -1*np.ones(numrows,int);
        numc = self.n_cluster
        random.seed()
        kpoints = np.ones_like(xdata_ml[0:numc,:],dtype=int)
        for i in range(0,numc):
            while True:
                index = random.randint(0,numc-1)
                kpoints[i]=xdata_ml[index]
                earlyBreak = False
                for j in range(0,i):
                    if LA.norm(kpoints[i]-kpoints[j]) == 0:
                        earlyBreak = True
                        break
                if not earlyBreak:
                    break
        distance = np.zeros(numc,float)
        distance[:]=float('inf')
        for i in range(0,numrows):
            tmp=np.apply_along_axis(LA.norm,1,kpoints-xdata_ml[i])
            ytrain[i]=np.argmin(tmp)
        if np.size(np.unique(ytrain),0) != numc:
            raise ValueError('Ah initial failed to have %d clusters'%numc)
        return ytrain



        

    """
    This is very delicate
    """
    def inverse_transform(self,xdata_ml):
        numrows = np.size(xdata_ml,0)
        if(len(xdata_ml.shape) > 1):
            featIndex_t=np.tile(self.featIndex,(numrows,1))
            xdata_nz = (xdata_ml == 1)
            for j in self.unique_feats:
                xdata_nz[:,j]=True
            
            res = np.extract(xdata_nz,featIndex_t).reshape(numrows,-1)
            return res
        else:
            xdata_ml2=np.atleast_2d(xdata_ml)
            for j in self.unique_feats:
                xdata_ml2[0][j]=1
            featIndex_t=np.atleast_2d(self.featIndex)
            return featIndex_t[:,np.nonzero(xdata_ml2)[1]]


    """
    Parameters:
        jll: 
            type: numpy array; shape: [nclass_,nbinaryfeature_]
        featArray:
            type: list; format: for each row of jll, item indexes from featArray[i] to featArray[i+1](exclusive) is 
            the binary result of fiture i
    Return:
        LCT table:
            type: ndarray; shape: [nclass_,n_oriFeature,max_nClass]
    """
    def calCPT(self):
        jll = self.feature_log_prob_
        featArray = self.nfeatures

        nClass  =np.size(jll,0)
        nFeature=np.size(jll,1)

        ori_nFeature=len(featArray)-1
        if self._verbose:    
            print "nFeature: %d; nClass: %d; ori_nFeature: %d"%(nFeature,nClass,ori_nFeature)
        if ori_nFeature < 1 or nFeature != featArray[-1]:
            raise ValueError("the dimension of given jll: %d * %d is inconsistent with info of featArray: %s!"%(nClass, nFeature,str(featArray)))
            return None

        nfeatArray=featArray-np.roll(featArray,1)
        max_nClass=np.amax(nfeatArray[1:])
        self.cpt=np.zeros((nClass,ori_nFeature,max_nClass))
        for i in range(0,nClass):
            for j in range(0,ori_nFeature):
                sumij=logsumexp(jll[i,featArray[j]:featArray[j+1]])
                for k in range(featArray[j],featArray[j+1]):
                    self.cpt[i,j,k-featArray[j]]=jll[i,k]-sumij
        self.cpt=np.exp(self.cpt)

    def outputCPT(self,out=None):
        if out == None:
            out=sys.stdout
        if self.cpt == None or self.xkeys == None:
            raise ValueError("OH I don't have conditional probability table or attributes information")
            return None

        nFeature = len(self.xkeys)
        curnumc     = np.size(self.cpt,0)
        for i in range(0,nFeature):
            title=self.featnames[i]
            for j in range(0,curnumc):
                title+=',class %d'%j
            print >>out,title
            for j in range(0,len(self.xkeys[i])):
                title=self.xkeys[i][j]
                for k in range(0,curnumc):
                    frac = self.cpt[k,i,j]
                    title+=",%f"%frac
                print >>out,title
            print >> out,"" 


    """
    xdata_ml is the transformed xdata
    ydata is the index version of original label information
    """
    def testModel(self,xdata_ml,ydata,timestamp=True,OUTPUTDIR=None):
        if self.ykeys == None:
            self.ykeys = range(0,np.amax(ydata))
        ykeys= self.ykeys

        if OUTPUTDIR == None:
            if self.outputDir != None:
                OUTPUTDIR = self.outputDir
            else:
                print "Ah not call self.setoutput yet. I will use current directory"
                OUTPUTDIR = "./"

        #curnumc = len(self.classes_)
        curnumc = np.size(self.feature_log_prob_,0)
        numrows = np.size(xdata_ml,0)
        ypredict = self.predict(xdata_ml)
        numc = len(ykeys)
        #ykeys,garbage,ydata_n = np.unique(xdata[:,k],True,True)
        dist = np.zeros((curnumc,numc))
        for i in range(0,curnumc):
            a=(ypredict==i)
            for j in range(0,numc):
                oj = (ydata == j)
                dist[i][j]=np.sum(np.multiply(a,oj))

        self.printStats(dist)
        self.outputCPT()

        outputDate=strftime("%m%d%H%M%S",localtime())
        prefix='test_nbem'
        if timestamp:
            outname="%s_i%d_r%d_n%d_k%d_%s.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
            outname_hu="%s_i%d_r%d_n%d_k%d_%s_hu.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
        else:
            outname="%s_i%d_r%d_n%d_k%d.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster)
            outname_hu="%s_i%d_r%d_n%d_k%d_hu.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster)

        out=open(os.path.join(OUTPUTDIR,outname),'w')
        out_hu=open(os.path.join(OUTPUTDIR,outname_hu),'w')

        title = ""
        for attr in self.featnames:
            title +="%s,"%attr
        title_hu=title

        title+='predicted_class,numerical_class'
        title_hu+='class,predicted_class,numerical_class'

        print >> out,title
        print >> out_hu,title_hu
        nFeats = len(self.featnames)
        xdata_ori = self.inverse_transform(xdata_ml)
        for i in range(0,numrows):
            onerow=""
            onerow_hu=""
            for j in range(0,len(xdata_ori[i])):
                item = xdata_ori[i][j]
                onerow+="%d,"%item
                onerow_hu+="%s,"%self.xkeys[j][item]
            onerow+="%d,%d"%(ypredict[i],ydata[i])
            onerow_hu+="%s,%d,%d"%(str(ykeys[ydata[i]]),ypredict[i],ydata[i])

            print >> out,onerow
            print >> out_hu,onerow_hu 

        out.close()
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
        print >>out_hu,""
        self.outputCPT(out_hu)
        print >>out_hu,""
        self.printStats(dist,out_hu)
        out_hu.close()

    def printStats(self,dist,out=None):
        
        if out==None:
            out=sys.stdout

        lct = self.cpt

        print >>out, ""
        print >>out, "statistics of naive bayes model"
        print >>out, "number of class: %d; number of features: %d"%(np.size(lct,0),np.size(lct,1))
        ykeys = self.ykeys

        numc = len(ykeys)
        for i in range (0,numc):
            print >>out, "%s ==> class %d"%(ykeys[i],i)
        
        curnumc = np.size(lct,0)

        print >>out,""
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

        print >>out,""

    

class MultinomialNBEM(BaseMultinomialNBEM):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        BaseMultinomialNBEM.__init__(self,alpha,fit_prior,class_prior)

    def build(self,n_cluster,xtrain,iterSN=10,iterCN=100,initMethod=0,timestamp=False,ydata=None):
        if n_cluster <=1:
            raise ValueError("Please input a maximum cluster number no smaller than 1")
        if iterCN<0:
            raise ValueError("Please input a strict positive integer value for iteration number of EM method")
        if iterSN<=0:
            raise ValueError("Please input a strict positive integer value for retrial number of EM method")

        self.n_cluster=n_cluster
        self.iterCN = iterCN
        self.iterSN = iterSN
        self.init =initMethod

        numrows = np.size(xtrain,0)
        numc = self.n_cluster
        if self._verbose:
            print "NO_Class,NO_Trial,NO_ITER,LL,DIFF_CPT,YET_CUR_BEST_LL,Comments"

        if self.outputDir!=None:
            prefix="log_nbem"
            outputDate=strftime("%m%d%H%M%S",localtime())
            if timestamp:
                logname="%s_i%d_r%d_n%d_k%d_%s.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
            else:
                logname="%s_i%d_r%d_n%d_k%d.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster)

            log=open(os.path.join(self.outputDir,logname),'w')    
            print >>log,"NO_Class,NO_Trial,NO_ITER,LL,DIFF_LL,DIFF_CPT,YET_CUR_BEST_LL,Comments"

        bestlog_prob = -float('inf') 
        best_iter = 0
        best_class_prior = None 
        best_feature_log_prob = None
        for j in range(0,self.iterSN):
        #Initializing step of target
            if initMethod == 0:
                ytrain=self.uniform_init_from_data(xtrain)
                if ydata!=None:
                    ytrain=ydata
                self.fit(xtrain,ytrain,class_prior=self.class_prior);
            elif initMethod == 1:
                ytrain=self.k_points_init(xtrain)
                if ydata!=None:
                    ytrain=ydata
                self.fit(xtrain,ytrain,class_prior=self.class_prior);
            elif initMethod == 2:
                #The following 2 lines take no effect in fact
                ytrain=self.k_points_init(xtrain)
                self.fit(xtrain,ytrain,class_prior=self.class_prior);

                self.uniform_init_theta(xtrain)
            elif initMethod == 3:
                ytrain=self.kmeans_init(xtrain)
                if ydata!=None:
                    ytrain=ydata
                self.fit(xtrain,ytrain,class_prior=self.class_prior);
            else:
                raise ValueError("Ah I don't know this initial method: %d"%initMethod)
        #initial
            old_sigma_yx=np.array(np.zeros((numrows,numc)),float)
            diff = 10000.0
            old_log_prob = 0.0
            for i in range(0,iterCN):
            #E-step
                sigma_yx=self.predict_proba(xtrain)
                diff_sig=sigma_yx-old_sigma_yx
                diff=LA.norm(diff_sig)
                old_sigma_yx=sigma_yx
            #M-step
                #q_y=np.sum(sigma_yx,axis=0)/numrows 
                q_y = np.sum(sigma_yx,axis=0)+self.alpha-1
                q = np.sum(q_y)
                q_y = np.divide(q_y,q) 
                self.class_log_prior_=np.log(q_y)

                #alpha is very import to smooth. or else in log when the proba is too small we got -inf
                #ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha
                ######MAP###########################################
                ncx = safe_sparse_dot(sigma_yx.T, xtrain)+self.alpha-1
                ncxsum=np.sum(ncx,axis=1)
                qxy=np.divide(ncx.T,ncxsum).T
                self.feature_log_prob_=np.log(qxy)
# I am stopped here
                if self.outputDir or self._verbose:
                    if i%10 ==0 or i > self.iterCN-5:
                    #if i < self.iterCN:
                        log_prob=self.calcObj(xtrain)
                        if self._verbose:
                            print "%d,%d,%d,%f,%f,%f,%f,Still in CN Loop"%(numc,j+1,i+1,log_prob,log_prob-old_log_prob,diff,bestlog_prob)
                        if self.outputDir:
                            print >>log,"%d,%d,%d,%f,%f,%f,%f,Still in CN Loop"%(numc,j+1,i+1,log_prob,log_prob-old_log_prob,diff,bestlog_prob)

                        old_log_prob = log_prob


            final_log_prob = self.calcObj(xtrain)

            if final_log_prob > bestlog_prob:
                if self.outputDir or self._verbose:
                    if self._verbose:
                        print "%d,%d,%d,%f,%f,%f,Better LL and NO Conflict"%(numc,j+1,iterCN,final_log_prob,diff,bestlog_prob)
                    if self.outputDir:
                        print >>log,"%d,%d,%d,%f,%f,%f,Better LL and NO Conflict"%(numc,j+1,iterCN,final_log_prob,diff,bestlog_prob)

                bestlog_prob = final_log_prob
                best_iter = j
                best_class_log_prior=copy.deepcopy(self.class_log_prior_)
                best_feature_log_prob=copy.deepcopy(self.feature_log_prob_)

                        

        self.class_log_prior_=copy.deepcopy(best_class_log_prior)
        self.feature_log_prob_=copy.deepcopy(best_feature_log_prob)

        print "Best one is at %dth iteration"%best_iter
        print "The corresponding log_prob: ", bestlog_prob

        if self.outputDir:
            print >>log,"Best one is at %dth iteration"%best_iter
            print >>log,"The corresponding log_prob: ", bestlog_prob
            log.close()

        self.calCPT()
        print "BIC: %f"%self.BIC(xtrain)


    """
    Attention: xtrain should be in well transformed format
    """
    def calcObj(self,xtrain,obj='MAP'):
        jll = self._joint_log_likelihood(xtrain)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        log_prob = np.sum(log_prob_x,axis=0)
        if obj == 'ML':
            return log_prob
        elif obj == 'MAP':
            log_theta = np.sum(self.class_log_prior_)+np.sum(self.feature_log_prob_)
            log_prob = log_prob+(self.alpha-1)*log_theta
            return log_prob

    def BIC(self,xtest):
        ll=self.calcObj(xtest,obj='ML')
        return ll-0.5*(self.n_cluster-1+self.n_cluster*(self.nfeatures[-1]-len(self.nfeatures)+1))*np.log(np.size(xtest,0))

    def get_sufficient_stats(self,xtest):
        sigma_yx=self.predict_proba(xtest)
        self.expect_nclass=np.sum(sigma_yx,axis=0)

        
    def complete_ML(self):


    def maximum_likelihood_c(self,xtest):
        logP_D1_M
        logP_D1_theta_M
        logP_D_theta_M
        return logP_D1_M-logP_D1_theta_M+logP_D_theta_M




