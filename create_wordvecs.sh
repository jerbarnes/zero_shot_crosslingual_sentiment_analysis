#!/bin/bash

#This script creates the word2vec models that are used in the experiment

langs=(en es)
types=(cbow skipgram)
sizes=(100 200 300)
word2vec_dir=$1

for lang in ${langs[*]}; do

    for type in ${types[*]}; do
       
	for size in ${sizes[*]}; do
	
	    if [[ $type == "cbow" ]]; then
		cbow=1
	    else
		cbow=0
	    fi

	    $word2vec_dir/word2vec -train data/corpus.1.$lang -output data/vecs/mt_data/"$lang"_"$type"_"$size"_w5_sample1e-5_hs0_min5.txt -size $size -sample 1e-5 -hs 0 -window 5 -threads 4 -cbow $cbow

	done
    done
done
