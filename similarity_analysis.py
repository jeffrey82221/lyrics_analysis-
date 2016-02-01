
# XXX read data from file ##########################################################
import numpy as np
song_info = [element.split('\t') for element in [line.rstrip('\n') for line in open('data/Western_songs_info.tsv')]]
song_ids = [element for element in [line.split(',') for line in open('data/Western_songs.csv')][0]]
song_lyrics_tmp = [element.split('\t') for element in open('data/Western_songs_lyrics.tsv')]
song_lyrics_data = [[element[0],element[1].split('\\n')] for element in song_lyrics_tmp]
song_info
song_lyrics_data

# import class that can fatch the lyrics and song data
from ReadInfo import SongInfo,SongInfoData,SentenceInfo,LyricsInfo,LyricsData

#REVIEW: #### initialize the lyrics_data object from database
lyrics_data = LyricsData(song_lyrics_data)
song_info_data = SongInfoData(song_info)
# XXX form an voc list with voc id
voc_dict = lyrics_data.dict_generate()
lyrics_data.indexify()
len(song_info_data.ids)

#REVIEW:####load the reduction 2D embedding###########################
import pickle
pfile = open("outkk_c1_d64_walk_10_tsne_d2",'r')
embedding_2D = pickle.load(pfile)
pfile.close()

#XXX Load the un-reduce-dimensionalized embedding
result_lines = [line.rstrip('\n') for line in open('kk_c1_d64_walk_10.embeddings')]
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
embedding_matrix = np.matrix(embedding_array)

#XXX check if two embedding matrix are in same data size
np.shape(embedding_2D)
np.shape(embedding_matrix)


#TODO Build similarity matrix

embedding_2D
np.shape(embedding_matrix)

voc_size = len(lyrics_data.voc_dict[1])
from scipy import sparse,spatial

song_matrix = embedding_matrix[voc_size:,:]
voc_matrix = embedding_matrix[:voc_size,:]

voc_kdtree = spatial.KDTree(voc_matrix)
song_kdtree = spatial.KDTree(song_matrix)
def pair_result(query_result,lyrics_data,song_info_data,SorV):
    result = []
    voc_size = len(lyrics_data.voc_dict[1])
    for element in query_result:
        if SorV=='v':
            result.append(lyrics_data.voc_dict[1][element])
        elif SorV == 's':
            try:
                ids = np.sort(lyrics_data.ids)
                si = song_info_data.findInfobyID(ids[element])
                result.append((si.title,si.album,si.artist))
            except:
                result.append(("none","none","none"))
    return result


ids = np.sort(lyrics_data.ids)

##REVIEW Searching voc by song
song_index = 1

print "search voc by song : "
song_info_data.findInfobyID(ids[song_index]).print_info()


_,voc_result = voc_kdtree.query(song_matrix[song_index],10)

print pair_result(voc_result[0],lyrics_data,song_info_data,'v')


voc_index = lyrics_data.voc_dict[0]["fuck"]
##REVIEW Searching song by voc
print "search song by voc : "
print lyrics_data.voc_dict[1][voc_index]
_,song_result = song_kdtree.query(voc_matrix[voc_index],10)


for element in pair_result(song_result[0],lyrics_data,song_info_data,'s'):
    print "title : ",element[0]
    print "album : ",element[1]
    print "artist :",element[2]
    print "----------------------------------------------------"

##REVIEW Searching voc by voc
voc_index = lyrics_data.voc_dict[0]["sex"]
print "search voc by voc : "
print lyrics_data.voc_dict[1][voc_index]
_,voc_result = voc_kdtree.query(voc_matrix[voc_index],10)

print pair_result(voc_result[0],lyrics_data,song_info_data,'v')

##REVIEW Searching song by song
song_index = 10

print "search song by song : "
song_info_data.findInfobyID(ids[song_index]).print_info()
_,song_result = song_kdtree.query(song_matrix[song_index],10)



for element in pair_result(song_result[0],lyrics_data,song_info_data,'s'):
    print "title : ",element[0]
    print "album : ",element[1]
    print "artist :",element[2]
    print "----------------------------------------------------"





#NOTE: Look out that the ids in lyrics_data are not sorted, but the embedding result are sorted with ids !!
