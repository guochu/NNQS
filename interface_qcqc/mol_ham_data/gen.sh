#!/bin/bash
set -x

# sh gen.sh 140 1.40
#target=n2_$1
target=f2
#target=h2_$1
#target=lih_$1
mkdir $target
cp template/gen_ham.py $target 
cd $target
python=/szcs/software/python/python3.7.2/bin/python3.7
$python gen_ham.py $2 | tee log
