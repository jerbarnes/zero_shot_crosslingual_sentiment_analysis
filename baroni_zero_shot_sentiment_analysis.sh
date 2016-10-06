#!/bin/bash


for english_file in $(ls data/vecs/mt_data/en*); do
    
    for spanish_file in $(ls data/vecs/mt_data/es*); do
    
	
	python3 zero_shot_baroni.py $spanish_file $english_file >> results.txt

    done
done