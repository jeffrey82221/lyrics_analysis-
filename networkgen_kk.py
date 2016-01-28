



####NLTK Testing ##############################################################################
from nltk import *
# from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from numpy import *
import matplotlib.pyplot as plt
sb = SnowballStemmer('english')
# lc = LancasterStemmer()


def stem_tokens(tokens, stemmer):
    # TODO we can write our own stemming method !
    # stemming function is executed here
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # NOTE here the tokenize method are the default method and the stemming
    # method can be alter !
    tokens = word_tokenize(text)
    #stems = stem_tokens(tokens, sb)
    stems = tokens
    return stems
################################################################################


##### XXX MAIN CODE START FROM HERE !
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

#REVIEW:####loading the object#########################
import pickle
fileh2 = open("lyricsdata",'r')
lyrics_data = pickle.load(fileh2)
fileh2.close()

#REVIEW:####saving the object###########################
import pickle
filehandler = open("lyricsdata",'w')
pickle.dump(lyrics_data,filehandler)
filehandler.close()
filehandler = open("songinfodata",'w')
pickle.dump(song_info_data,filehandler)
filehandler.close()
########################################################
#XXX Read in the Deepwalk generated embedding file
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
embedding_2D = model.fit_transform(embedding_matrix)




#TESTING : ####################################################################
#lyrics_data.ids
#song_info_data.ids
#song_info_data.findInfobyID(lyrics_data.ids[4]).print_info()
import random
index = int(np.floor(random.uniform(0,len(lyrics_data.lyricsinfos))))
lyrics_data.lyricsinfos[index].print_lyrics()
lyrics_data.lyricsinfos[index].ID
song_info_data.findInfobyID(lyrics_data.lyricsinfos[index].ID).print_info()
print song_lyrics_data[index][1][4]
lyrics_data.lyricsinfos[2100].sentenceInfos[1].tokenized_sentences
lyrics_data.lyricsinfos[2100].voc_set()

song_info_data.findInfobyID(lyrics_data.lyricsinfos[128].ID).print_info()


lyrics_data.lyricsinfos[157].voc_set()
voc_array.index(lyrics_data.lyricsinfos[157].sentenceInfos[1].tokenized_sentences[0])
voc_array[6850]






# TODO create an adj matrix of song id and voc id

# 14883 lyrics data provided !
# lyrics_data_ids = [int(element) for element in lyrics_data.ids] #574723~63476142

len(lyrics_data.ids)
len(song_info_data.ids)
lyrics_data.lyricsinfos[0].voc_set()
voc_dict[1].keys()
lyrics_data.lyricsinfos[1].voc_set()
adjlist = []
for element in lyrics_data.lyricsinfos:
    adjlist.extend(zip([element.ID]*len(element.voc_set()),list(element.voc_set())))

# generate adjlist file

adjlist
outfile = open("inkk.adjlist",'w')
for element in adjlist:
    outfile.write(str(element[0]))
    outfile.write(" ")
    outfile.write(str(element[1]))
    outfile.write("\n")

outfile.close()

# TODO create an adj matrix of song id and voc id with the same sentence connected


# Future TODO :  divided the lyrics by there paragraph and identify their type of paragraph! It can be an additional feature (HOW?)
# NOTE:有些歌詞不一定單純以starttime == 0 和 endTime == 0 (簡稱00) 的數量來分界，有可能是以很多個連續的00來分界！

paragraph_num = []
for lyrics in lyrics_data.lyricsinfos:
    ar = np.array([element.startTime==0 and element.endTime==0 for element in lyrics.sentenceInfos])
    num = np.array(ar[1:])-np.array(ar[:-1])
    paragraph_num.append(sum(num))


paragraph_num = [sum([element.startTime==0 and element.endTime==0 for element in lyrics.sentenceInfos]) for lyrics in lyrics_data.lyricsinfos]

sentenece_num = [len(lyrics.sentenceInfos) for lyrics in lyrics_data.lyricsinfos]

##ploting and testing
import matplotlib.pyplot as plt

plt.scatter(paragraph_num,sentenece_num)
max(paragraph_num)
lyrics_data.lyricsinfos[argmax(paragraph_num)].print_lyrics()
min(paragraph_num)
max(sentenece_num)
min(sentenece_num)

plt.show()
