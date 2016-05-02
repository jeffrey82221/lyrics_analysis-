# NOTE: In this code, it will generate an edglist containing song and
# lyrics relationship.

# Pre-requirement files :
# 1. 'data/Western_songs_info.tsv' : contain the meta data of each song and their corresponding song id
# 2. 'data/Western_songs.csv' : contain the title of each song and their corresponding song id
# 3. 'data/Western_songs_lyrics.tsv' : contain the lyrics of each song. More , the lyrics are with alignment of the audio using time indexes.
# OUTPUT :
# 1. cleaned.adjlist : In the edgelist (adjlist), every index in the edge list correspond to a song node (song ID) or a vocabrary node (smaller voc index).
# If a song node is connected to a voc node , it means that the song contain the voc in its lyrics.
# If a voc node is connected to a song node, it means that the voc is contained in that song's lyrics.
# 2. voc.dict : the file containing dictionary of the whole vocabrary, the
# first maps term to index, the second maps index to term.


# XXX MAIN CODE START FROM HERE !
from __future__ import print_function
# XXX read data from file ################################################
from detecting_language import *
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing.pool import ApplyResult

song_info = [element.split('\t') for element in [line.rstrip(
    '\n') for line in open('data/Western_songs_info.tsv')]]
song_ids = [element for element in [line.split(
    ',') for line in open('data/Western_songs.csv')][0]]
song_lyrics_tmp = [element.split('\t') for element in open(
    'data/Western_songs_lyrics.tsv')]
song_lyrics_data = [[element[0], element[1].split(
    '\\n')] for element in song_lyrics_tmp]

print('number of songs in Western_songs.csv', len(song_ids))
print('number of songs in Western_songs_info.csv', len(song_info))
print('number of songs in Western_songs_lyrics.tsv', len(song_lyrics_tmp))
# TODO:remove none english songs :
# detecting language
#%timeit language_tag = [get_language(''.join([sen for sen in element[1]])) for element in song_lyrics_data]
# REVIEW original: 43.7s parallelize: 30.9 s


def detect_language_tag(element):
    return get_language(''.join([sen for sen in element[1]]))

language_tag = Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(detect_language_tag)(element) for element in song_lyrics_data)


new_song_lyrics_data = [song_lyrics_data[i] for i in range(
    len(language_tag)) if language_tag[i] == 'english']
print('song number after non english songs are removed:', len(new_song_lyrics_data))

##########################################################################


##########################################################################
# import class that can fatch the lyrics and song data
import nltk
from joblib import Parallel, delayed
import multiprocessing
from ReadInfo import *

# REVIEW: #### initialize the lyrics_data object from database
# TODO:tokenization is still not very accurate!!
# TODO:remove the sentences without words
lyrics_data = LyricsData(new_song_lyrics_data)
song_info_data = SongInfoData(song_info)

# XXX form an voc list with voc id
voc_dict = lyrics_data.dict_generate()

print('voc size of tokenzied terms:', len(voc_dict[0]))


# store voc_dict to file
import pickle
f = open('dict.voc', 'w')
pickle.dump(voc_dict, f, pickle.HIGHEST_PROTOCOL)


##########################################################################
# TODO spelling check on voc_list "spelling_check.py"

##########################################################################


# XXX create an adj matrix of song id and voc id

# generate voc_set_index for each lyrics
def adjlist_generation(l):
    voc_index_set = map(lambda x: voc_dict[0][x], l.voc_set())
    return zip([l.ID] * len(voc_index_set), list(voc_index_set))

lyrics_adjlist = Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(adjlist_generation)(l) for l in lyrics_data.lyricsinfos)


adjlist = []
for a in lyrics_adjlist:
    adjlist.extend(a)

# 14883 lyrics data provided !
# lyrics_data_ids = [int(element) for element in lyrics_data.ids]
# #574723~63476142

# generate adjlist file


outfile = open("cleaned.adjlist", 'w')
for element in adjlist:
    outfile.write(str(element[0]))
    outfile.write(" ")
    outfile.write(str(element[1]))
    outfile.write("\n")
outfile.close()

##########################################################################
# TODO create an adj matrix of song id and voc id with the same sentence
# connected
