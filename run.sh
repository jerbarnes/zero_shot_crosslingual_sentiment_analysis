#!/bin/bash

word_to_vec_dir=/home/jeremy/NS/Keep/Permanent/Tools/word2vec-master/bin

bash create_wordvecs.sh $word_to_vec_dir

bash baroni_zero_shot_sentiment_analysis.sh