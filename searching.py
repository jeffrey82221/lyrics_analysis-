#reading from particular embedding
#XXX Load the un reduce-dimensionalized embedding
result_lines = [line.rstrip('\n') for line in open('embeddings/kk_c1_d64_walk_100_cleaned.embeddings')]
len(result_lines)

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
embedding_matrix = np.array(embedding_array)

from scipy import spatial

#initialize parameters
sorted_ids=sorted_ids.tolist()
song_embedding = embedding_matrix[voc_size:,:] #TODO change to un-dimension-reducted embedding

#customized parameters
search_count = 50
query_song = song_embedding[sorted_ids.index(1674621)]
#query_song = song_embedding['ids?']

#searching algorithm
searched_indexs = spatial.KDTree(song_embedding).query(query_song,k=search_count)[1]

searched_ids = [sorted_ids[index] for index in searched_indexs]

#print result ids
searched_ids

#plot
import matplotlib.pyplot as plt

distances = spatial.KDTree(song_embedding).query(query_song,k=search_count)[0]
plt.plot(distances)
plt.show()
