#! /bin/bash
i=1
names=(`echo $2|tr '/' '\n'`)
basename=(`echo ${names[-1]}|tr '\.' '\n'`)
basename=${basename[0]}
if [ -d $basename ];then
    echo "$basename exists! Renamed"
    mv -f $basename ${basename}_old
fi
mkdir $basename
cstable=${basename}/${basename}_cstable
echo "Cheeseman Stutz Scores of different models" > $cstable
s=20
n=150
while [ $i -le $1 ];
do
    echo "./test_naive_bayes_EM.py -i 1 -c 0 -n $n -s $s -k $i -odu $2 >> ${basename}/${basename}_log$i"
    ./test_naive_bayes_EM.py -i 1 -c 0 -n $n -s $s -k $i -oud $2 >> ${basename}/${basename}_log$i
    echo "$i clusters" >> $cstable
    head -n 5 ${basename}/${basename}_log$i>> $cstable
    echo "" >> $cstable
    i=$(($i+1))
done

