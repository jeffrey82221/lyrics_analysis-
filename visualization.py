from __future__ import print_function

#REVIEW:####load the reduction 2D embedding###########################
import numpy as np
keys = np.loadtxt('out_new_2d.embeddings.keys').astype(int)
voc_keys = keys[:31214]
song_keys = keys[31214:]
em = np.loadtxt('out_new_2d.embeddings')

#TODO input song title, artist, album for each song
song_info = [element.split('\t') for element in [line.rstrip('\n')
                                                 for line in open('data/Western_songs_info.tsv')]]

from ReadInfo import *
song_info_data = SongInfoData(song_info)
artist_list = [song_info_data.findInfobyID(sk).artist for sk in song_keys]
title_list = [song_info_data.findInfobyID(sk).title for sk in song_keys]
album_list = [song_info_data.findInfobyID(sk).album for sk in song_keys]
#TODO match each voc index with its voc word




#TODO visualization using bokeh
