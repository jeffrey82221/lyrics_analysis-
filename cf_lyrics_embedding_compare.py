
from __future__ import print_function

import sys

filename = 'win7.w100.l32.d128.embeddings'
#TODO: input song-voc embedding (*.embeddings)

import numpy as np
result_lines = [line.rstrip('\n') for line in open(filename)]
object_count = len(result_lines)
splited_result_lines = []
for i in range(1, len(result_lines)):
    splited_result_lines.append(result_lines[i].split())

len(splited_result_lines)
embedding_list = []
for items in splited_result_lines:
    embedding_list.append(
        (int(items[0]), [float(item) for item in items[1:]]))


embedding_list.sort()

embedding_key = [e[0] for e in embedding_list]
embedding_array = [e[1] for e in embedding_list]
#TODO: input song cf embeddings (*.csv)
#TODO: select the matched song embedding out of the two embedding
#TODO: seperate the song embedding into training set ,validation set and testing set
#TODO: construct a neural network with song-voc embedding as input and song cf embedding as output
#TODO: create a function for generating batch in every training iteration
