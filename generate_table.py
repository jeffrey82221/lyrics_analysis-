from __future__ import print_function
import sys
try:
    filename = sys.argv[1]

except:
    print("no enought argument input ! \n should input :\n 1. input filename")
    exit()

import numpy as np
import pickle
import numpy as np
import pickle
voc_dict=pickle.load(open('dict.voc','rb'))
keys = np.loadtxt(filename+'.keys').astype(int)
voc_size = len(voc_dict[1])
voc_keys = keys[:voc_size]
song_keys = keys[voc_size:]
lyrics_size = len(song_keys)
em = np.loadtxt(filename)
embedding = np.matrix(em)
dim = embedding.shape[1]
#import meta data
#TODO input song title, artist, album for each song
song_info = [element.split('\t') for element in [line.rstrip('\n')
                                                 for line in open('data/Western_songs_info.tsv')]]

from ReadInfo import *
song_info_data = SongInfoData(song_info)

artist_list = [song_info_data.findInfobyID(sk).artist for sk in song_keys]
title_list = [song_info_data.findInfobyID(sk).title for sk in song_keys]
album_list = [song_info_data.findInfobyID(sk).album for sk in song_keys]

#TODO match each voc index with its voc word
voc_list = [voc_dict[1][e] for e in voc_keys]

from pandas import DataFrame

song_dict = dict()
song_dict['title']=title_list
song_dict['album']=album_list
song_dict['artist']=artist_list
voc_dict = dict()
voc_dict['voc']=voc_list
for i in range(dim):
    song_dict['d'+str(i)]=em[voc_size:,i]
    voc_dict['d'+str(i)]=em[:voc_size,i]
song_df=DataFrame(song_dict)
voc_df=DataFrame(voc_dict)

song_df.to_excel(filename+'.song.xlsx', sheet_name='sheet1', index=False)
voc_df.to_excel(filename+'.voc.xlsx', sheet_name='sheet1', index=False)
