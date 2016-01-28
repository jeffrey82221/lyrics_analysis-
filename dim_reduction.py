import numpy as np
result_lines = [line.rstrip('\n') for line in open('outkk.embeddings')]
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
from sklearn.manifold import TSNE
model = TSNE(n_components=2, perplexity=30.0, early_exaggeration=10.0, learning_rate=1000.0,
     n_iter=200, metric='euclidean', init='random')

print "start transforming..."
embedding_2D = model.fit_transform(embedding_matrix)


print "start saving the result..."
#REVIEW:####saving the object###########################
import pickle
pfile = open("reduct_embedding",'w')
pickle.dump(embedding_2D,pfile)
pfile.close()
