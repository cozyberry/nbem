#! /usr/bin/python
## @brief Documentation of one example how we use MultinomialNBEM class to train naive bayesian network with unlabeled data. True label information can also be provided to offer more methods for validation of the model.
#@package test_nbem
#@author Wei HU
#@version 2.0
#@note This script shows how to train a model with given number of clusters. User then can try different specific number of clusters and stick to a particular one after comparing marginal likelihood for each model. Here we provide 2 types of scores: BIC or Cheeseman_Stuzt score. They are both approximations of marginal likelihood. According to previous studies, Cheeseman_Stutz score seems to be more accurate"
#

import naive_bayes_EM
import os,sys,getopt,random,csv
import numpy as np
from optparse import OptionParser, OptionGroup
import argparse


##Default value for max number of iterations for one EM round
ITERCN = 20

##Default value for number of restarts of EM rounds
ITERSN = 1

##Default output directory for output files
OUTPUTDIR =os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'outputs/')

##A function for parsing different options.
#    Type python test_nbem.py -h for detailed information
#    Parses command line options.
#    Generally we're  providing diverse options for training a naive bayesian network.
#    Please be patient, don't panic with these many options. They do exist for a reason LOL
#    Type easily "python test_nbem.py -h" to display help information
#
def argParse():

    global ITERCN
    global ITERSN
    global OUTPUTDIR

    parser = argparse.ArgumentParser(description='An example of training a naive bayesian network with a given number of clusters',prog=os.path.basename(sys.argv[0]),formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('datafilename', metavar='datafilename', type=str, nargs=1,
                   help='Input data file name for feeding the training engine')
    parser.add_argument(
        "-k", "--kcluster",
        action="store", type=int, dest="numc",required=True,
        help="Specify number of clusters for this training task. One training process is done with specific number of existing clusters. Then users can try with different specified number of clusters and stick to a particular one after comparing marginal likelihood for each model. Here we provide 2 types of approximate scores: BIC or Cheeseman_Stuzt score.\n\n"
    )
    parser.add_argument(
        "-n",
        action="store", type=int, dest="itercn",default=ITERCN,
        help="Specify max number of times in each round of EM iteration process.\n\n"
    )
    parser.add_argument(
        "-r",
        action="store", type=int, dest="itersn",default=ITERSN,
        help="Specify number of times of restarts with a different initial configuration of the network.\n\n"
    )
    parser.add_argument(
        "-i", "--initmethod",
        action="store", type=int, dest="initMethod",default=0,
        help="""Specify method for obtaining an initial parameter configuration for the naive bayes network.\n0: obtain the initial parameter values by labeling each datum randomly;\n1: obtain the initial parameter values by an initial k-means clustering;\n2: obtain an initial paramter configuration by choosing from paramter allowed space uniformely;\n3: obtain the initial parameter values by usered defined label information.\n\n"""
    )
    parser.add_argument(
        "--alpha",
        action="store", type=float, dest="alpha",default=2.0,
        help="Specify alpha value for network parameters'prior dirichelet distribution\n\n"
    )
    parser.add_argument(
        "-a", "--attributes",
        action="store_true", dest="_attr",default=False,
        help="Provide data file with first line of attributes information. By default the training engine eats data with no such information.\n\n"
    )
    parser.add_argument(
        "-l", "--labled",
        action="store_true", dest="_labeled",default=False,
        help="Provide data file with last column of true label information. This label information won't be used for training unless the -i option set at 3. Normally it will be used in later test process to provide more insights of clustering quality.\n\n"
    )
    parser.add_argument(
        "-o", "--outputdir",
        action="store", type=str, dest="outputdir",default=OUTPUTDIR,
        help="Specify a output directory for model outputs.\n\n"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true", dest="_verbose",default=False,
        help="Triggle verbose mode for displaying detail training information each time.\n\n"
    )
    parser.add_argument(
        "-t", "--timestamp",
        action="store_true", dest="_timestamp",default=False,
        help="Timestamp each output file to avoid overwriting outputs during different training process. In this case each output file would have a suffix of the type '%%m%%d%%H%%M%%S' e.g. 0807113024, which means generated at August 7th 11h:30m:24s.\n\n"
    )
    args = parser.parse_args()
    try:
        args.datafilename=args.datafilename[0]
        if args.initMethod > 3 or args.initMethod < 0:
            raise ValueError('Error: Unexpected value for initial configuration method. Accepted method set: [0,1,2,3]')
        if args.itercn < 0 or args.itersn < 0:
            raise ValueError('Error: Positie value expected for max number of iteration and number of restarts')
        if args.numc < 1 :
            raise ValueError('Error: Number of given clusters should be at least 1')
        if not (os.path.exists(args.datafilename) and os.path.isfile(args.datafilename)):
            raise IOError('Error: Input data file: %s not found!'%args.datafilename)
        if not (os.path.exists(args.outputdir) and os.path.isdir(args.outputdir)):
            os.mkdir(args.outputdir)
            print "Remarks: Output directory :%s does not exist. I have created one for you."%args.outputdir

        return args

    except ValueError as e:
        print e
        print "Please type python test_nbem.py -h for usage information"
    except IOError as e:
        print e
        print "Please type python test_nbem.py -h for usage information"
    except OSError as e:
        print e
        print "Please type python test_nbem.py -h for usage information"

    sys.exit(1)

## @brief A function which read a csv format file
#
#    Retrieves rows from a given csv file.
#
#
# @param		filename	Given sample file. Each row is comme separated representing one sample datum. Each column represents a feature. If provided with true label information, it should be put as the last column
# @param		_attr	Specify whether provided with first line as titles of each feature
#
# @return
#        A numpy array representation of given data. Size = [n_sample,n_feature] (or [n_sample,n_feature+1] when label information exists)
#
#
# @exception		IOError	An error occured accssing the given input data file
# @exception		ValueError	An error occured when given invalid datafile. A valid data should consist same dimensional datum. But it is ok that there exists empty feature name when attributes infomation provided as well.
#
def getcsvData(filename,_attr=False):
    if not (os.path.exists(filename) and os.path.isfile(filename)):
        raise IOError("Error: File %s not found!"%filename)

    """
    Open the inputfile and count nonempty rows.
    Moreover test whether each nonempty row has the same dimension.
    A valid data should consist same dimensional  nonempty datum. But it is ok that there exists empty feature name when attributes infomation provided as well.
    """
    numrows = 0
    ncols = 0
    ct = 0
    attributes=[]
    csvreader=csv.reader(open(filename,'r'))
    for row in csvreader:
        ct+=1
        if _attr and ct==1:
            attributes=row
            pass
        elif len(row)!=0:
            numrows+=1
            if numrows == 1:
                ncols = len(row)
            elif ncols != len(row):
                raise ValueError("Error: inconsistent column number at line: %d of file: %s"%(ct,filename))

    """Read data in numpy array"""
    csvreader=csv.reader(open(filename,'r'))
    data=np.ndarray(shape=(numrows,ncols), dtype=object)
    ct=0
    k=0
    for row in csvreader:
        ct+=1
        if _attr and ct==1:
            pass
        elif len(row)!=0:
            data[k,:]=np.array(row,dtype=str)
            k+=1
    return data,attributes



## @brief Main function for training a naive bayesian network using unlabeled data by EM method
#
# @b     Outputs:
#        If particular output directory is provided after -o option, all the output files would be stored there. Or ele by default they will be stored in outputs/ directory in the current directory of this scripts
#        After running, we will get 4 output files. They are named in the following fashion:
#
#        log_nbem_ix_rx_nx.csv
#        score_nbem_ix_rx_nx_kx.csv
#        test_nbem_hu_ix_rx_nx_kx.csv
#        test_nbem_ix_rx_nx_kx.csv (where the x represents user input option value, i.e. i ==> initial Method; r ==> restarts number; n==> max number of iteration during one iteration. If -t option is triggered, all these filenames will have a timestamp suffix in addition)
#        'log_nbem_ix_rx_nx.csv' consists info with the evolution of score during each EM round.
#        'score_nbem_ix_rx_nx_kx.csv' contains final evaluated score of current model and the final parameter configuration.
#        'test_nbem_hu_ix_rx_nx_kx.csv' is simply the original csv file plus predicted class attribute by the model
#        'test_nbem_hu_ix_rx_nx_kx.csv' is index based version of 'test_nbem_hu_ix_rx_nx_kx'. In this file the value of each feature is in numerated fashion
#
#
def main():

    args=argParse()

    """Initialize training data"""
    traindata,attributes = getcsvData(args.datafilename,args._attr)

    """
    Create a MultinomialNBEM object, which represents one bayesian network learned from unlabeled data
    """
    nbem=naive_bayes_EM.MultinomialNBEM(alpha=args.alpha)
    nbem.setVerbose(args._verbose)
    nbem.setOutput(args.outputdir)

    """
    Preprocessing original raw data.
    This preprocessing includes: 1)transformation original raw data into index based data;2) binarization of returned index based data.
    For example, if the raw data is (array([[2, 2],
                                            [2, 3]])
    After transformation step 1), we got (array([[0 0],
                                                 [0 1]])
    Since for column 0, there exists 1 unique value, and for column 1 there exists 2 different values.
    After transformation step 2), we further got array([[1, 1, 0],
                                                        [1, 0, 1]])
    Please Note that I binarized each column, which means that for column 0, [0, ==> [1,
                                                                              0]      1]
                                                               for column 1, [0, ==> [1,0],
                                                                              1]     [0,1]
    It is easy to see that in fact after binarization the position of the occurrence '1' indicates the multi value
    For details please refet ro naive_bayes_EM's implementations.
    """
    if args._labeled:
        xdata_ml,ydata=nbem.fit_transformRaw(traindata,_labeled=args._labeled,arrayAttr=attributes)
    else:
        xdata_ml=nbem.fit_transformRaw(traindata,_labeled=args._labeled,arrayAttr=attributes)

    """
    Learn MAP Parameters through EM method
    For details please refer to naive_bayes_EM's documentation
    """
    nbem.build(args.numc,xdata_ml,args.itersn,args.itercn,args.initMethod,args._timestamp)

    """
    Test the model with original training data.
    """
    if args._labeled:
        nbem.testModel(xdata_ml,ydata,args._timestamp)
    else:
        nbem.testModel(xdata_ml,timestamp=args._timestamp)

if __name__=='__main__':
    main()
