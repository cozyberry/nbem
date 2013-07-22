#! /bin/bash
i=1
names=(`echo $2|tr '/' '\n'`)
basename=(`echo ${names[-1]}|tr '\.' '\n'`)
basename=${basename[0]}
mkdir $basename
cstable=${basename}/${basename}_cstable
echo "Cheeseman Stutz Scores of different models" > $cstable
while [ $i -le $1 ];
do
    echo "./test_naive_bayes_EM.py -i 1 -c 0 -n 100 -s 10 -k $i -od $2 >> ${basename}/${basename}_log$i"
    ./test_naive_bayes_EM.py -i 1 -c 0 -n 100 -s 10 -k $i -od $2 >> ${basename}/${basename}_log$i
    echo "$i clusters" >> $cstable
    head -n 4 ${basename}/${basename}_log$i>> $cstable
    echo "" >> $cstable
    i=$(($i+1))
done

