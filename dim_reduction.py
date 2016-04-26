import sys
print sys.argv

try:
    inputfilename = sys.argv[0]
    outputfilename = sys.argv[1]
    dimension = sys.argv[2]
except:
    print "no enought argument input , should input 1. input filename 2. output filename 3. dimension"




import numpy as np
result_lines = [line.rstrip('\n') for line in open(inputfilename)]
result_lines

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
from tsne import bh_sne
#model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=10.0, learning_rate=1000.0,  n_iter=200, metric='euclidean', init='random')

print "start transforming..."
#embedding_2D = model.fit_transform(embedding_matrix)
embedding_2D = bh_sne(embedding_matrix,d=dimension,theta=0.2)

#TODO:save the result as excel file

print "start saving the result..."
#REVIEW:####saving the object###########################
import pickle
pfile = open(outputfilename,'w')
pickle.dump(embedding_2D,pfile)
pfile.close()
