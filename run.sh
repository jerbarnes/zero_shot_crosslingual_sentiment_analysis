#!/bin/bash

#First set word_to_vec_path to your local word2vec-master/bin path
word_to_vec_path= my_word_to_vec_path

bash create_wordvecs.sh $word_to_vec_path

bash baroni_zero_shot_sentiment_analysis.sh
