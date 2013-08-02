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
        now=`date +%m%d%H%M%S`
        mv  ${basename} ${basename}$now
    fi
else mkdir $basename
fi
cstable=${basename}/${basename}_cstable
echo "Cheeseman Stutz Scores of different models" > $cstable
s=20
n=150
while [ $i -le $1 ];
do
    #echo "./test_naive_bayes_EM.py -i 1 -c 0 -n $n -s $s -k $i -od $2 >> ${basename}/${basename}_log$i"
    #./test_naive_bayes_EM.py -i 1 -c 0 -n $n -s $s -k $i -od $2 >> ${basename}/${basename}_log$i
    cmd="./test_naive_bayes_EM.py -i 1 -c 0 -n $n -s $s -k $i -ou $2 > ${basename}/${basename}_log$i"
    echo $cmd
    eval $cmd
    echo "$i clusters" >> $cstable
    head -n 5 ${basename}/${basename}_log$i>> $cstable
    echo "" >> $cstable
    i=$(($i+1))
done

