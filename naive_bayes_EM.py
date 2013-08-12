#! /usr/bin/python
## @package naive_bayes_EM
#  naive_bayes implements training of a Naive Bayes Network from unlabeled data. I use class structure of naive_bayes.BaseDiscreteNB which offers a quite good way of how to store the paremeter configuration of a naive bayes net. From this structure naive_bayes_EM itself implements a CategoricalNB class in which all the nodes follow a Categorical distribution and a CategoricalNBEM class which uses EM method to learn MAP parameter configuration. Details of implemenation are given in CategoricalNBEM.build() function.
#
#  The basic idea of training is that: given a dataset \f${\mathbf{X_1},\mathbf{X_2},...,\mathbf{X_N}}\f$ where N is the number of samples and each  we would like to train a naive bayes network. And \f$ \mathbf{X_r} = {x_1,x_2,..,x_n}\f$where n is the number of features. Given a particular number of cluster k, our engine aimes at finding a MAP parameter configuration \f$ \theta_{MAP}  = argmax_{\theta}P(\theta|D,m)\f$, where \f$m\f$ refers to the structure of segmentation mode, more precisely, a naive bayesian network with latent class node as the root having k number of possible clustering result. For more detailed mathematical introduction of model specification, please refer to my report as well as these papers Cheeseman1995 @cite acsbayesian Chickering1997 @cite chickering1997efficient and Friedman1997@cite friedman1997bayesian
#
#  @note  The reason why I did not use the MultinomialNB directly is that the MultinomialNB supposes that each feature follows a multinomial distribution, in our training task, each feature follows a categorical distibution. This means for for feature \f$x_i\f$ which has \f$r_i \f$ possible state of value, \f$ {P(x_i=v_j|c) = p_j, j\in {1,2,...,r_i}}\f$ . However Categorical distribution is just the generization of Bernoulli distribution. So it is quite natual to reuse BaseDiscreteNB for my class.
#
#  @warning The difference between multinomial distribution and categorical distribution determines how we should preprocessing the given data. The original implementation of naive_bayes.MultinomialNB is specially used for text classification. In the source code MultinomialNB did binarization for label data(y), not for feature data(xdata). Forcategorical distribution we should be both binarization operation for xdata as well as for y. Please see CategoricalNB.fit_transformRaw() function for details of implementation.
#
#  @remark
#  Please go to http://scikit-learn.org/stable/modules/naive_bayes.html for detailed information on used naive_bayes models.
#  And http://thinkmodelcode.blogspot.fr/2013/04/naive-bayes-classification-using-python.html for an example of classification task. The classification task is more simple since it does not require EM method.
#

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import cluster
import numpy as np
from numpy import linalg as LA
from sklearn.utils.extmath import safe_sparse_dot, logsumexp
from sklearn.utils import array2d, atleast2d_or_csr
import os,sys
import string
from time import localtime, strftime, time
import itertools
import random
import copy
import pprint
from scipy import special
from scipy.sparse import issparse

## @brief     a Multinomial Naive Bayes Cluster which combines [naive bayes classifier implementation] and [EM or ECM method] to deal with missing label information.
#
#    The Multinomial Naive Bayes Cluster is suitable for clustering with
#    discrete features (e.g., word counts for text classification).
#
#    The Multinomial Naive Bayes Cluster can accept data without or with label information.
#    But the label information would only be used as informative guides
#    For one naive bayes network \f$m\f$, each local distribution function \f$p(x_i|\mathbf{pa_i},\mathbf{\theta_m},\mathbf{m})\f$ is consists a set of categorical distribution. That is for each \f$i,j\f$
#    Parameters
#    ----------
# 	alpha	float, optional (default=1.0)
#        Additive (Laplace/Lidstone) smoothing parameter
#        (0 for no smoothing).
#
# 	fit_prior	boolean
#        Whether to learn class prior probabilities or not.
#        If false, a uniform prior will be used.
#
# 	class_prior	array-like, size=[n_classes,]
#        Prior probabilities of the classes. If specified the priors are not
#        adjusted according to the data.
#
# 	iterSN	iteration number for EM
# 	iterCN	retrial number for EM
#
#    Attributes
#    ----------
#
# 	n_cluster	initial maximum number of cluster
#
# 	local_prob_table	local probability table for each feature node. shape=[n_classes,n_features]
#
# 	featIndex	an internal array index for multi-value features. shape=[sum of domain length of each feature]
#
# 	nfeatures	an internal array of domain length of each multi-value feature shape=[feature number+1]
#
#    `intercept_`, `class_log_prior_` : array, shape = [n_classes]
#        Smoothed empirical log probability for each class.
#
#    `feature_log_prob_`, `coef_` : array, shape = [n_classes, n_features]
#        Empirical log probability of features
#        given a class, P(x_i|y).
#
#        (`intercept_` and `coef_` are properties
#        referring to `class_log_prior_` and
#        `feature_log_prob_`, respectively.)
#
#    Examples
#    --------
#    To be modified
#    >>> import numpy as np
#    >>> X = np.random.randint(5, size=(6, 100))
#    >>> Y = np.array([1, 2, 3, 4, 5, 6])
#    >>> from sklearn.naive_bayes import MultinomialNB
#    >>> clf = MultinomialNB()
#    >>> clf.fit(X, Y)
#    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#    >>> print(clf.predict(X[2]))
#    [3]
#
#    Notes
#    -----
#    To be modified
#    For the rationale behind the names `coef_` and `intercept_`, i.e.
#    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
#    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
#
class CategoricalNB(naive_bayes.BaseDiscreteNB):



    ## @brief Constructor of CategoricalNB class
    #
    # @param		alpha	float, optional, default=1.0
    #                Parameter in the Dirichelet prior distribution of all the parameters. In bayesian statistique, we have prior distribution for parameters. A general Dirichlet distribution looks like this: \f$f(x_1,\dots, x_{K-1}; \alpha_1,\dots, \alpha_K) = \frac{1}{\mathrm{B}(\alpha)} \prod_{i=1}^K x_i^{\alpha_i - 1}\f$ for all \f$x_1,\dots,x_{K} > 0 \f$ satisfying \f$x_1 + \dots + x_{K} = 1 \f$. Here we use a symmetric Dirichlet distribution, where all of the elements making up the parameter vector \f$\alpha\f$ have the same value "alpha" in the input parameter lists. Intuitively a given \f$\alpha\f$ vector means adding \f$\alpha_i\f$ imaginary occurrences of event \f$x_i\f$. The reason why we choose Dirichelet distribution is because it is the conjugate prior of Categorical distribution and Multinomial Distribution.
    # @param		fit_prior	boolean, optional, default=True
    #                Whether to learn class prior probabilities or not.
    #                If false, a uniform prior will be used.
    # @param		class_prior	array-like, size=[n_classes,],default=None
    #                Prior probabilities of the classes. If specified the priors are not
    #                adjusted according to the data.
    #
    #        @see http://en.wikipedia.org/wiki/Dirichlet_distribution for details of Dirichelet distribution. And Friedman1997@cite friedman1997bayesian for details of prior distribution and why we choose dirichelet distribution as the prior.
    #
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self._verbose = False
        self.outputDir = None

        if fit_prior == True and class_prior:
            """Confilicts of given arguments"""
            print "Warning: the fit_prior and class_prior are both set. We will use the assigned class_prior and the fit_prior will be ignored"

    ## @brief set verbose mode by the input argument
    def setVerbose(self,verbose):
        self._verbose = verbose

    ## @brief set output directory for engine outputs
    def setOutput(self,outputdir):
        if (not os.path.exists(outputdir)) or (not os.path.isdir(outputdir)):
            raise ValueError("The output directory is invalide %s"%outputdir)
            return False
        self.outputDir = outputdir

    def _count(self, X, Y):
            """Count and smooth feature occurrences."""
            if np.any((X.data if issparse(X) else X) < 0):
                raise ValueError("Input X must be non-negative.")
            N_c_i = safe_sparse_dot(Y.T, X) + self.alpha-1#
            N_c = np.sum(N_c_i, axis=1)

            return N_c, N_c_i

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        X = atleast2d_or_csr(X)
        return (safe_sparse_dot(X, self.feature_log_prob_.T)
                + self.class_log_prior_)





    """
    if _labeled is True, the last column of data should be label information
    data is in the form of raw data, e.g. high,2,cheap,vgood
    Transform multi-value xdata to binary representation of xdata_ml.
    For example, for a feature with 4 possible value: bad,good,vgood, the binary representation of a value 'bad' or 'good' or 'vgood' would be [1,0,0],[0,1,0] or [0,0,1] respectively
    """
    def fit_transformRaw(self,data,_labeled=True,arrayAttr=[]):
        self._labeled=_labeled

        if len(arrayAttr)!=0:
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
                    cur_xdata_ml = 1-cur_xdata_ml
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
                index = random.randint(0,numrows-1)
                kpoints[i]=xdata_ml[index,:]
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

    def get_filename(self,prefix,timestamp=True):

        outputDate=strftime("%m%d%H%M%S",localtime())
        if timestamp:
                    outname="%s_i%d_r%d_n%d_k%d_%s.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
        else:
                    outname="%s_i%d_r%d_n%d_k%d.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster)
        return outname





    """
    xdata_ml is the transformed xdata
    ydata is the index version of original label information
    """
    def testModel(self,xdata_ml,ydata=None,timestamp=True,OUTPUTDIR=None):

        if ydata!=None:
            if self.ykeys == None:
                self.ykeys = range(0,np.amax(ydata))
            ykeys= self.ykeys
            numc = len(ykeys)

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
        #ykeys,garbage,ydata_n = np.unique(xdata[:,k],True,True)
        if ydata!=None:
            dist = np.zeros((curnumc,numc))
            for i in range(0,curnumc):
                a=(ypredict==i)
                for j in range(0,numc):
                    oj = (ydata == j)
                    dist[i][j]=np.sum(np.multiply(a,oj))

            self.printStats(dist)

        self.outputCPT()

        outname=self.get_filename('test_nbem',timestamp)
        outname_hu=self.get_filename('test_nbem_hu',timestamp)
        #if timestamp:
            #outname="%s_i%d_r%d_n%d_k%d_%s.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
            #outname_hu="%s_i%d_r%d_n%d_k%d_%s_hu.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
            #score_filename="%s_i%d_r%d_n%d_k%d_%s.csv"%(prefix1,self.init,self.iterSN,self.iterCN,self.n_cluster,outputDate)
        #else:
            #outname="%s_i%d_r%d_n%d_k%d.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster)
            #outname_hu="%s_i%d_r%d_n%d_k%d_hu.csv"%(prefix,self.init,self.iterSN,self.iterCN,self.n_cluster)
            #score_filename="%s_i%d_r%d_n%d_k%d.csv"%(prefix1,self.init,self.iterSN,self.iterCN,self.n_cluster)

        out=open(os.path.join(OUTPUTDIR,outname),'w')
        out_hu=open(os.path.join(OUTPUTDIR,outname_hu),'w')

        title = ""
        for attr in self.featnames:
            title +="%s,"%attr
        title_hu=title
        if ydata != None:
            title+='predicted_class,numerical_class'
            title_hu+='class,predicted_class,numerical_class'
        else:
            title+='predicted_class'
            title_hu+='predicted_class'


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
            if ydata!=None:
                onerow+="%d,%d"%(ypredict[i],ydata[i])
                onerow_hu+="%s,%d,%d"%(str(ykeys[ydata[i]]),ypredict[i],ydata[i])
            else:
                onerow+="%d"%ypredict[i]
                onerow_hu+="%d"%ypredict[i]


            print >> out,onerow
            print >> out_hu,onerow_hu

        out.close()
        print >>out_hu,""
        if ydata!=None:
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
        if ydata!=None:
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



class CategoricalNBEM(CategoricalNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        CategoricalNB.__init__(self,alpha,fit_prior,class_prior)


    def build(self,n_cluster,xtrain,iterSN=10,iterCN=100,initMethod=0,timestamp=False,ydata=None,_bayes=False):
        if n_cluster <1:
            raise ValueError("Please input a maximum cluster number no smaller than 1")
        if iterCN<0:
            raise ValueError("Please input a strict positive integer value for iteration number of EM method")
        if iterSN<=0:
            raise ValueError("Please input a strict positive integer value for retrial number of EM method")

        self.n_cluster=n_cluster
        self.iterCN = iterCN
        self.iterSN = iterSN
        self.init =initMethod
        self._bayes = _bayes

        self.d=self.n_cluster-1+self.n_cluster*(self.nfeatures[-1]-len(self.nfeatures)+1)

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
        classes_backup = None
        if numc == 1:
            self.iterSN = 1

        for j in range(0,self.iterSN):
            if numc == 1:
                self.classes_=np.array([1],int)
                sigma_yx=np.ones((numrows,1),float)
                q_y = np.sum(sigma_yx,axis=0)+self.alpha-1#
                q = np.sum(q_y)
                q_y = np.divide(q_y,q)
                self.class_log_prior_=np.log(q_y)
                ncx = safe_sparse_dot(sigma_yx.T, xtrain)+self.alpha-1#
                ncxsum=np.sum(ncx,axis=1)
                qxy=np.divide(ncx.T,ncxsum).T
                self.feature_log_prob_=np.log(qxy)
                self.iterCN = 0
                initMethod = -1
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
            elif initMethod == -1:
                pass
            else:
                raise ValueError("Ah I don't know this initial method: %d"%initMethod)
        #initial difference
            old_sigma_yx=np.array(np.zeros((numrows,numc)),float)
            diff = 10000.0
            old_log_prob = 0.0
            for i in range(0,self.iterCN):
            #E-step
                sigma_yx=self.predict_proba(xtrain)
                diff_sig=sigma_yx-old_sigma_yx
                diff=LA.norm(diff_sig)
                old_sigma_yx=sigma_yx
            #M-step
                #q_y=np.sum(sigma_yx,axis=0)/numrows
                q_y = np.sum(sigma_yx,axis=0)+self.alpha-1#
                q = np.sum(q_y)
                q_y = np.divide(q_y,q)
                self.class_log_prior_=np.log(q_y)

                #alpha is very import to smooth. or else in log when the proba is too small we got -inf
                #ncx = safe_sparse_dot(sigma_yx.T, xtrain)+mnb.alpha-1
                ######MAP###########################################
                ncx = safe_sparse_dot(sigma_yx.T, xtrain)+self.alpha-1#
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
                classes_backup = copy.deepcopy(self.classes_)



        self.class_log_prior_=copy.deepcopy(best_class_log_prior)
        self.feature_log_prob_=copy.deepcopy(best_feature_log_prob)
        self.classes_=copy.deepcopy(classes_backup)

        if self._bayes:
            ytrain=self.predict(xtrain)
            naive_bayes.MultinomialNB.fit(self,xtrain,ytrain)
            print "Bayes smoothing..."

        print "Best one is at %dth iteration"%best_iter

        if self.outputDir:
            print >>log,"Best one is at %dth iteration"%best_iter
            print >>log,"The corresponding log_prob: ", bestlog_prob
            log.close()

        self.calCPT()
        #print "%d clusters; %d params"%(self.n_cluster,self.d)
        #print "BIC: %f"%self.BIC(xtrain)
        #print "Cheeseman_Stutz_Score: %0.15f"%self.cheeseman_stutz_score(xtrain)
        print "n_cluster,Log_MAP,BIC,CS_Marginal_likelihood"
        print "%f,%f,%f,%f"%(self.n_cluster,bestlog_prob,self.BIC(xtrain),self.cheeseman_stutz_score(xtrain))

        score_filename=self.get_filename('score_nbem',timestamp=timestamp)
        score_filename=os.path.join(self.outputDir,score_filename)
        score_file=open(score_filename,'w')
        print >>score_file,"%d clusters; %d params"%(self.n_cluster,self.d)
        print >>score_file,"BIC: %f"%self.BIC(xtrain)
        print >>score_file,"Cheeseman_Stutz_Score: %0.15f"%self.cheeseman_stutz_score(xtrain)
        print >>score_file,""
        self.outputCPT(score_file)
        print >>score_file,""
        line = ""
        for i in range(0,len(self.class_log_prior_)):
            line +=",class%d"%i
        print >>score_file,line
        line=""
        for i in range(0,len(self.class_log_prior_)):
            line+=",%f"%np.exp(self.class_log_prior_[i])

        print >>score_file,line


        score_file.close()

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
            log_prob = log_prob+(self.alpha-1)*log_theta#
            return log_prob

    def BIC(self,xtest):
        ll=self.calcObj(xtest,obj='ML')
        return ll-0.5*self.d*np.log(np.size(xtest,0))


    def get_sufficient_stats(self,xtest):
        sigma_yx = self.predict_proba(xtest)
        self.expect_nclass = np.sum(sigma_yx,axis=0)
        self.expect_nclass_feature = safe_sparse_dot(sigma_yx.T, xtest)
        """
        for i in range(0,len(self.nfeatures)-1):
            if i == 0:
                self.sum_expect_nclass_feature = np.sum(self.expect_nclass_feature[:,self.nfeatures[i]:self.nfeatures[i+1]],axis=1)
            else:
                self.sum_expect_nclass_feature = np.hstack((self.sum_expect_nclass_feature,np.sum(self.expect_nclass_feature[:,self.nfeatures[i]:self.nfeatures[i+1]],axis=1)))
        print np.sum(self.expect_nclass)
        print self.sum_expect_nclass_feature
        print self.expect_nclass
        print self.expect_nclass_feature
        """

    ## Calculate the prior probability of one parameter configuration,\f$\mathbf{\theta_{m}}\f$.
    #
    #  As we explained in CategoricalNB, we have a collection of parameter set \f$\mathbf{\theta_{ij}}\f$ representing the distribution of node \f$i\f$'values when its parent takes value \f$j\f$. we have \f$\mathbf{\theta_{ij}} = \biggl(\theta_{ij2},\dots,\theta_{ijr_i}\biggr)\f$ and \f$ \sum_{k=1}^{r_i}{\theta_{ijk}} = 1, \theta_{ijk} > 0 \f$. In our model we use symmetric Dirichelet distribution as \f$ \mathbf{\theta_{ij}} \f$ 's prior distribution
    #
    #  Equation of prior probability of one parameter configuration, \f$ \mathbf{\theta_{m}}\f$: \f{eqnarray}{p(\mathbf{\theta_{m}}|\mathbf{m})=\prod_{i=1}^{n}{\prod_{j=1}^{q_i}{p(\mathbf{\theta_{ij}}|\mathbf{m})=}}\f}
    #
    def complete_model_ML(self):
        gamma_nclass=special.gammaln(self.expect_nclass+self.alpha)
        gamma_nclass_feature=special.gammaln(self.expect_nclass_feature+self.alpha)
        gamma_sum_nclass=special.gammaln(np.sum(self.expect_nclass+self.alpha))
        nFeature=len(self.nfeatures)-1
        for i in range(0,nFeature):
            cur_gamma_sum_nclass_feature=special.gammaln(np.sum(self.expect_nclass_feature[:,self.nfeatures[i]:self.nfeatures[i+1]]+self.alpha,axis=1))
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

    ## @brief Calculate Marginal likelihood of the data when the data is completed given the model and the parameter. We use the sufficient statistics corresponding to the given MAP parameter configuration.
    #
    #  Equation of logarithm of Marginal Likelihood:\f{eqnarray}{
    #\log{p(D|m)} &=& \log{\prod_{i=1}^{n}{\prod_{j=1}^{q_i}{\biggl(\frac{\Gamma(\alpha_{ij})}{\Gamma(\alpha_{ij}+N_{ij})}\prod_{k=1}^{r_i}{\frac{\Gamma(\alpha_{ijk}+N_{ijk})}{\Gamma(\alpha_{ijk})}}\Biggr)}}}\\
    #&=& \sum_{i=1}^{n}{\sum_{j=1}^{q_i}{\Biggl(\log{\frac{\Gamma(\alpha_{ij})}{\Gamma(\alpha_{ij}+N_{ij})}}+\sum_{k=1}^{r_i}{\log{\frac{\Gamma(\alpha_{ijk}+N_{ijk})}{\Gamma(\alpha_{ijk})}}}\Biggr)}}\f}
    #  Here since we use a symmetric Dirichlet distribution, we have \f$ \alpha_{ijk} = \alpha \f$ for all \f$i,j,k\f$ and \f$\alpha_{ij} = \sum_{k=1}^{r_i}{\alpha_{ijk}} \f$. And the collection of \f$N_{ijk}\f$ are sufficient statistics of the data for the model \f$m\f$. \f$N_{ij} = \sum_{k=1}^{r_i}{N_{ijk}}\f$.In EM process we use the expected value of these sufficient statistics during this calculation. Please go to CategoricalNBEM.build() for more details.@see Chickering197@cite chickering1997efficient for details of formule deduction.
    #  @return \f$\log{p(D|m)}\f$
    #  @note the sufficient statistics require label information as well, here what we use is in fact the expectations of sufficient statistics. 
    #
    def complete_param_ML(self):
        part1=np.multiply(self.expect_nclass,self.class_log_prior_)
        part2=np.multiply(self.expect_nclass_feature,self.feature_log_prob_)
        return np.sum(part1)+np.sum(part2)
        #return np.multiply(self.expect_nclass,self.class_log_prior_)+np.multiply(self.expect_nclass_feature,self.feature_log_prob_)




    ## @brief Calculate approximate marginal likelihood by cheeseman stutz method.This method is proposed initially in Cheeseman1985. Equation of CS_Score: \f{eqnarray}{ \log{p(D|m)} \approx \log{p(D^{'}|m)} - \log{p(D^{'}|\tilde{\phi}_m,m)} + \log{p(D|\tilde{\phi}_m,m)}\f} where \f$ D^{'}\f$ is the extended dataset, that is, the original data set \f$D\f$ plus label info. We use such an extended dataset that its sufficient statistcs equals to those used for MAP configuration \f$\theta_MAP\f$(Cheeseman1985@cite acsbayesian, Chickering197@cite chickering1997efficient)
    #
    #  @param		xdata	data used to training the model.
    #  @see Cheeseman1985@cite acsbayesian, Chickering197@cite chickering1997efficient for details of formule deduction.
    #
    def cheeseman_stutz_score(self,xdata):

        ## Call CategoricalNBEM.get_sufficient_stats() to get expected sufficient statistics for MAP configuration \f$ \mathbf{\theta_{m}}: N_{ijk} \f$
        self.get_sufficient_stats(xdata)

        ## Calculate \f$ \log{p(D^{'}|m)} \f$
        logP_D1_M = self.complete_model_ML()

        ## Calculate \f$\log{p(D^{'}|\tilde{\phi}_m,m)}\f$
        logP_D1_theta_M = self.complete_param_ML()

        ## Calculate \f$\log{p(D|\tilde{\phi}_m,m)}\f$
        logP_D_theta_M = self.calcObj(xdata,obj='ML')

        return logP_D1_M-logP_D1_theta_M+logP_D_theta_M



