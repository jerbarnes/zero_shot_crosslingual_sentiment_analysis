########################################################
#Zero-shot learning for crosslingual sentiment analysis#
########################################################

Dependencies:
Python3 
Numpy
nltk
sklearn



#This experiment is based on Mikolov et al. (2013) Exploiting similarities between languages. The idea is to create a translation matrix that effectively maps between two monolingual vector representations, in essence translating in vector space. Here, we attempt to use this idea not for translation purposes, but as a way to enable crosslingual sentiment analysis. The performance, however, is not great. On the OpeNER Dataset, English-Spanish combination, 4-class classification, accuracy is around 48%.

Usage:

bash zero_shot_crosslingual_sentiment_analysis.sh

results will be kept in results.txt 
