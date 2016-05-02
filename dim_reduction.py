#NOTE:In this code, it convert a high dimension embeddings to a low dimension embedding using TSNE.
# Input :
# 1. the filename of the embedding file (.embeddings)
# 2. the dimension one wish to reducted into.
# Output :
# 1. the reducted voc and song embedding in file *.2d
# 2. the keys of the song and voc in file *.keys)



from __future__ import print_function

import sys

try:
    filename = sys.argv[1]
    dimension = sys.argv[2]
except:
    print("no enought argument input ! \n should input :\n 1. input filename  2. dimension")
    exit()

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
embedding_matrix = np.matrix(embedding_array)
# XXX T-SME visualization of result
#from sklearn.manifold import TSNE
import sklearn
def cosine_distance(X1,X2):
    return sklearn.metrics.pairwise.pairwise_distances(X1,X2, metric='cosine')

try:
    from tsne import bh_sne
    print("start transforming...")
    embedding_low = bh_sne(embedding_matrix,d=int(dimension),theta=0.2)
except:
    from sklearn.manifold import TSNE
    print("start transforming...")
    model = TSNE(n_components=int(dimension), perplexity=30.0, early_exaggeration=10.0, learning_rate=1000.0,  n_iter=1000, metric='euclidean', init='pca',angle=0.2)
    embedding_low = model.fit_transform(embedding_matrix)
#
#TODO:save the result as excel file
print("start saving the result...")
#REVIEW:####saving the object###########################
outputfilename=filename+'.'+str(dimension)+'d'
np.savetxt(outputfilename,embedding_low)
#save the mapping file (from order to index)
np.savetxt(outputfilename+'.keys',np.array(embedding_key).astype(int),fmt='%d')
print('saved file:',outputfilename,'size:',np.shape(embedding_low))
print('saved keys:',outputfilename+'.keys','size:',np.shape(np.array(embedding_key)))
#import pickle
#pfile = open(outputfilename,'w')
#pickle.dump(embedding_low,pfile)
#pfile.close()
