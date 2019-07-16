#!/bin/bash

FILES=./corpus/msr*.txt

for file in $FILES
do
    python ./src/preprocessing.py $file
done
