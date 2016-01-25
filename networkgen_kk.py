
# XXX read data from file
import numpy as np
song_info = [element.split('\t') for element in [line.rstrip('\n') for line in open('data/Western_songs_info.tsv')]]
song_ids = [element for element in [line.split(',') for line in open('data/Western_songs.csv')][0]]
song_lyrics_tmp = [element.split('\t') for element in open('data/Western_songs_lyrics.tsv')]
song_lyrics_data = [[element[0],element[1].split('\\n')] for element in song_lyrics_tmp]
song_info
song_lyrics_data

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


class SongInfo:
    ID = ""
    title = ""
    album = ""
    artist = ""
    def __init__(self,array):
        self.ID = array[0]
        self.title = array[1]
        self.album = array[2]
        self.artist = array[3]
    def print_info(self):
        print self.ID,",",self.title,",",self.album,",",self.artist
class SongInfoData:
    songinfos = []
    ids = []
    def __init__(self,array):
        for element in array:
            si = SongInfo(element)
            self.songinfos.append(si)
            self.ids.append(si.ID)
    def findInfobyID(self,id):
        return self.songinfos[self.ids.index(id)]
class SentenceInfo:
    startTime = 0
    endTime = 0
    sentenceType = 0
    sentence = ""
    tokenized_sentences = []
    def __init__(self,string):
        time_upper_bound = string.index('[')
        time_lower_bound = string.index(']')
        type_upper_bound = string.index('<')
        type_lower_bound = string.index('>')
        time_string = string[time_upper_bound+1:time_lower_bound]
        start_string,end_string = time_string.split(" ")
        self.startTime = float(start_string.split(':')[0])*60+float(start_string.split(':')[1])
        self.endTime = float(end_string.split(':')[0])*60+float(end_string.split(':')[1])
        self.sentenceType = int(string[type_lower_bound-1])
        self.sentence = sentence_cleaning(string[type_lower_bound+1:])
        self.tokenized_sentences = self.sentence.split(' ')
        return array

        for char in array:
            if ord(char)>=128:
                array = array.replace(char,'')
        return array
        if type==1:
            start = array.find( '(' )
            end = array.find( ')' )
        elif type==2:
            start = array.find( '[' )
            end = array.find( ']' )
        elif type==3:
            start = array.find( '<' )
            end = array.find( '>' )
        else:
            return array
        if start != -1 and end != -1:
            result = array[:start] + array[end+1:]
            return result
        else:
            return array
    def print_info(self):
        print self.startTime,",",self.endTime,",",self.sentenceType,",",self.sentence

class LyricsInfo:
    ID = ""
    sentenceInfos = []
    def __init__(self,array):
        self.ID = array[0]
        sentenceInfos = []
        for element in array[1][:-1]:
            try:
                sen_info = SentenceInfo(element)
                sentenceInfos.append(sen_info)
            except:
                sentenceInfos

        self.sentenceInfos = sentenceInfos
    def print_lyrics(self):
        for element in self.sentenceInfos:
            element.print_info()
    def voc_set(self):
        voc_set = set()
        for element in self.sentenceInfos:
            voc_set = set().union(*[voc_set,element.tokenized_sentences])
        return voc_set
class LyricsData:
    lyricsinfos = []
    ids = []
    def __init__(self,array):
        for element in array:
            li = LyricsInfo(element)
            self.lyricsinfos.append(li)
            self.ids.append(li.ID)
    # TODO: the function that can find an info of an corresponding lyrics
    def findInfobyID(self,id):
        return self.lyricsinfos[self.ids.index(id)]

    def voc_set(self):
        voc_set = set()
        for element in self.lyricsinfos:
            voc_set = set().union(*[voc_set,element.voc_set()])
        return voc_set



# TODO: What's the meaning of the types of the sentences?
# A.
# Meaning different singer
# Meaning not the lyrics sentence (introduction sentences)
# Meaning the same singer singing with different vocal voice ! (ex.different pitch, different timbre, hibrid voice...)
# Meaning rap!!


lyrics_data = LyricsData(song_lyrics_data)

song_info_data = SongInfoData(song_info)
#lyrics_data.ids
#song_info_data.ids
#song_info_data.findInfobyID(lyrics_data.ids[4]).print_info()
import random
index = int(np.floor(random.uniform(0,len(lyrics_data.lyricsinfos))))
lyrics_data.lyricsinfos[index].print_lyrics()
lyrics_data.lyricsinfos[2100].sentenceInfos[1].tokenized_sentences
lyrics_data.lyricsinfos[2100].voc_set()

song_info_data.findInfobyID(lyrics_data.lyricsinfos[128].ID).print_info()




voc_set = lyrics_data.lyricsinfos[157].voc_set()
voc_set = lyrics_data.voc_set()

# TODO tokenize each sentences + stem each sentences


# .lower
ex_string = lyrics_data.lyricsinfos[128].sentenceInfos[5].sentence

# Remove none ascii lyrics

# TODO : Before tokenize :
# remove the ascii letter


removenoneAscii(ex_string)
song_info_data[128]
lyrics_data.lyricsinfos[128].print_lyrics()
isAscii(lyrics_data.lyricsinfos[128].sentenceInfos[5].sentence)
lyrics_data.lyricsinfos[8].print_lyrics()

# remove the words in () and [] and <> eg. <sec_st>




removeparenthese("kk<abc>ttt",3)


# remove the senteces ends with "":


# remove the sentence with starttime = 0 and end time = 0


# TODO : divided the lyrics by there paragraph and identify their type of paragraph! It can be an additional feature (HOW?)
# NOTE:有些歌詞不一定單純以starttime == 0 和 endTime == 0 (簡稱00) 的數量來分界，有可能是以很多個連續的00來分界！

paragraph_num = []
for lyrics in lyrics_data.lyricsinfos:
    ar = np.array([element.startTime==0 and element.endTime==0 for element in lyrics.sentenceInfos])
    num = np.array(ar[1:])-np.array(ar[:-1])
    paragraph_num.append(sum(num))


paragraph_num = [sum([element.startTime==0 and element.endTime==0 for element in lyrics.sentenceInfos]) for lyrics in lyrics_data.lyricsinfos]

ex = [True,True,False,False,True]
np.array(ex[1:])-np.array(ex[:-1])
ar = np.array([element.startTime==0 and element.endTime==0 for element in lyrics_data.lyricsinfos[argmax(paragraph_num)].sentenceInfos])


plt.plot(np.logical_and(ar[:-1]==True, ar[1:]==False))





sentenece_num = [len(lyrics.sentenceInfos) for lyrics in lyrics_data.lyricsinfos]

import matplotlib.pyplot as plt

plt.scatter(paragraph_num,sentenece_num)
max(paragraph_num)
lyrics_data.lyricsinfos[argmax(paragraph_num)].print_lyrics()
min(paragraph_num)
max(sentenece_num)
min(sentenece_num)

plt.show()

# remove the adj sentenes with same starttime and same endtime #REVIEW
# set all uppercase to lower case







# TODO : After tokenize : (stemming process)
# remove the ... after or before words
# remove special character such as . , ! ? ... check by eyes #REVIEW
# recover the word with "-" inside it

# NOTE some special sentences needs to be remove
# if all character are not in Ascii letters
# word that includes [( on left or ]) on right
# sentence that contain no words
#  - it's OK
# remove the sentences that startTime is 0 and endTime is also 0
# sentence with same starttime and end time
# song adjecent sentences with same startitme and endtime : (might be title or have other functionality)
# sentence with : at the end might be some kind of title


# NOTE some special sentences needs to be alter
# all uppercase sentences
# remove the words in () or []
# remove the ... after some words, or before some words
# sometimes there are chinese tanslation cat after the original sentence (this kind of sentence should be clean particularly)
#   - remove the none ascii part
# sometimes the front sentences contain <sec_st> , TODO understand the meaning and try to clean it
# sometime the front sentences might be the introduction sentence (eg. with the titles and the artists)

# NOTE some spectial words in lyrics might need to be stem!
# words that contain lot of '-'
# words that are an abbreviation ex. I'm ,  thinkin' , He's , ...



song_lyrics_data
len(lyrics_data.ids)

lyrics_data.lyricsinfos[126].print_lyrics()



# TODO form an voc list with voc id


# TODO create an adj matrix of song id and voc id

# TODO create an adj matrix of song id and voc id with the same sentence connected




# TODO tokenize all lyrics
from nltk import *
from sklearn.feature_extraction.text import CountVectorizer
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
    stems = stem_tokens(tokens, lc)
    return stems

# vectorize each vector
count_vectorizer = CountVectorizer(tokenizer=tokenize)
count_vectorizer.fit_transform(lyrics)
len(count_vectorizer.vocabulary_)
feature_name = count_vectorizer.get_feature_names()
freq_term_matrix = count_vectorizer.transform(lyrics)
shape(freq_term_matrix.todense())

list(count_vectorizer.vocabulary_)
freq_term_matrix[0].todense()


# XXX change adjecent matrix to adjecent list
binary_adj_matrix = sign(freq_term_matrix.todense())

(song_count,voc_count)=shape(binary_adj_matrix)
# index song from 1 to 266, and words from 267 to 267 + 3999

adj_list = []

# find indexes that are k in a list
def find(k,lis):
    indexes = []
    for i in range(len(lis)):
        if lis[i]==k:
            indexes.append(i)
    return indexes
# generate adj list from 1 to 266

for i in range(song_count):
    adj_list.append((array(find(1,binary_adj_matrix[i].tolist()[0]))+song_count+1).tolist())

# generate adj list from 267~267+3999

for i in range(voc_count):
    adj_list.append((array(find(1,binary_adj_matrix.T[i].tolist()[0]))+1).tolist())

# print the adjlist
for i in range(song_count+voc_count):
    print(i+1,end=" ")
    for item in adj_list[i]:
        print(item,end=" ")
    print()


# print the adjlist into an file
outfile = open("in.adjlist",'w')
for i in range(song_count+voc_count):
    outfile.write(str(i+1))
    outfile.write(" ")
    for item in adj_list[i]:
        outfile.write(str(item))
        outfile.write(" ")
    outfile.write("\n")

outfile.close()

# print the vocabray into file
outfile = open("voc",'w')
for item in list(count_vectorizer.vocabulary_.keys()):
    outfile.write(item)
    outfile.write("\n")
outfile.close()

# TODO generate another kind of network structure ,
# such that word of each sentence are linked, and all words in each song are link to the song obect.
# NOTE it's more feasible to generate edge list !

tokenized_lyrics = [tokenize(item) for item in lyrics]
voc = set()

for item in tokenized_lyrics:
    voc = voc|set(item)


voc_ = list(voc)
voc_.sort()

voc_

#generate index list for each lyrics
indexed_lyrics = [[voc_.index(token)+len(lyrics)+1 for token in tokenized_lyric] for tokenized_lyric in tokenized_lyrics]

#generate edge list
# for every song-word-edge
song_word_edge = []

for i in range(len(lyrics)):
    song_word_edge.extend([(i+1,j) for j in indexed_lyrics[i]])
song_word_edge

# for every neighbor word pairs
word_word_edge = []

for index_lyric in indexed_lyrics:
    for i in range(len(index_lyric)-1):
        word_word_edge.append((index_lyric[i],index_lyric[i+1]))

all_edge_list = []
all_edge_list.extend(song_word_edge)
all_edge_list.extend(word_word_edge)

len(all_edge_list)
# print the edge list into file
all_edge_list
outfile = open("in2.adjlist",'w')
for item in all_edge_list:
    outfile.write(str(item[0]))
    outfile.write(" ")
    outfile.write(str(item[1]))
    outfile.write("\n")

outfile.close()

######END######
