#! /bin/bash
i=1
names=(`echo $2|tr '/' '\n'`)
basename=(`echo ${names[-1]}|tr '\.' '\n'`)
basename=${basename[0]}
if [ -d $basename ];then
    echo "$basename exists! Renamed. Do u want to remove it? (Y/y/N/n)"
    read ans
    if [ x$ans = xy ] || [ x$ans = xY ];then
        rm -r $basename
    else
        now=`date +%y%m%d%H%M%S`
        mv  ${basename} ${basename}$now
    fi
fi
mkdir $basename
cstable=${basename}/${basename}_cstable
s=64
n=50
while [ $i -le $1 ];
do
    outputdir=${basename}/cluster_$i
    mkdir $outputdir
    cmd="./test_naive_bayes_EM.py -i 1 -c 0 -n $n -s $s -k $i -o ${outputdir} -u $2 > ${basename}/${basename}_log$i"
    echo $cmd
    eval $cmd
    if [ $i -eq 1 ];then
        grep  -A1 "n_cluster,Log_MAP,BIC,CS_Marginal_likelihood" ${basename}/${basename}_log$i > $cstable
        #tail -n2 ${basename}/${basename}_log$i> $cstable
    else
        grep  -A1 "n_cluster,Log_MAP,BIC,CS_Marginal_likelihood" ${basename}/${basename}_log$i |tail -n1 >> $cstable
        #tail -n1 ${basename}/${basename}_log$i>> $cstable
    fi
    i=$(($i+1))
done

