

##### XXX MAIN CODE START FROM HERE !
# XXX read data from file ##########################################################
from detecting_language import *
import numpy as np
song_info = [element.split('\t') for element in [line.rstrip('\n') for line in open('data/Western_songs_info.tsv')]]
song_ids = [element for element in [line.split(',') for line in open('data/Western_songs.csv')][0]]
song_lyrics_tmp = [element.split('\t') for element in open('data/Western_songs_lyrics.tsv')]
song_lyrics_data = [[element[0],element[1].split('\\n')] for element in song_lyrics_tmp]

print 'number of songs in Western_songs.csv',len(song_ids)
print 'number of songs in Western_songs_info.csv',len(song_info)
print 'number of songs in Western_songs_lyrics.tsv',len(song_lyrics_tmp)
#TODO:remove none english songs :
#detecting language
language_tag = [get_language(''.join([sen for sen in element[1]])) for element in song_lyrics_data]

new_song_lyrics_data = [song_lyrics_data[i]  for i in range(len(language_tag)) if language_tag[i]=='english']
print 'song number after non english songs are removed:',len(new_song_lyrics_data)

##################################################################################


####################################################################################
# import class that can fatch the lyrics and song data
import nltk
from ReadInfo import *
#REVIEW: #### initialize the lyrics_data object from database
#TODO:tokenization is still not very accurate!!
#TODO:remove the sentences without words
lyrics_data = LyricsData(new_song_lyrics_data)

song_info_data = SongInfoData(song_info)

# XXX form an voc list with voc id
voc_dict = lyrics_data.dict_generate()
print 'voc size of tokenzied terms:',len(voc_dict[0])
lyrics_data.indexify()
####################################################################################
#TODO spelling check on voc_list "spelling_check.py"



###################################################################################
# TODO create an adj matrix of song id and voc id

# 14883 lyrics data provided !
# lyrics_data_ids = [int(element) for element in lyrics_data.ids] #574723~63476142




adjlist = []
for element in lyrics_data.lyricsinfos:
    adjlist.extend(zip([element.ID]*len(element.voc_set()),list(element.voc_set())))

# generate adjlist file


outfile = open("cleaned.adjlist",'w')
for element in adjlist:
    outfile.write(str(element[0]))
    outfile.write(" ")
    outfile.write(str(element[1]))
    outfile.write("\n")

outfile.close()
###################################################################################
# TODO create an adj matrix of song id and voc id with the same sentence connected