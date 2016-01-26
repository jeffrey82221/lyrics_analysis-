


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
    ID = None
    title = ""
    album = ""
    artist = ""
    def __init__(self,array):
        self.ID = int(array[0])
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


# TODO tokenize each sentences + stem each sentences


# TODO : Before tokenize :
# Remove none ascii letters
# remove words in parenthese
# set all uppercase to lower case (by .lower)

###Functions for cleaning the words in lyrics sentences

def removenoneAscii(array):
    for char in array:
        if ord(char)>=128:
            array = array.replace(char,'')
    return array
def removeparenthese(array,type):
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
def sentence_cleaning(array):
    array = removenoneAscii(array)
    array = removeparenthese(array,2)
    array = removeparenthese(array,3)
    return array.lower()



# TODO remove the sentences with following properties :
# remove the senteces ends with "" (with no abc in them):
# remove the sentences with starttime = 0 and end time = 0
# remove the adj sentenes with same starttime and same endtime #REVIEW
array = "abcSD_"


def noabc(array):
    if sum([element.isalpha() for element in array])==0:
        return True
    else:
        return False



# TODO : After tokenize : (stemming process)
# remove the ... after or before words
# remove special character such as . , ! ? ... check by eyes #REVIEW
# recover the word with "-" inside it


class SentenceInfo:
    startTime = 0
    endTime = 0
    sentenceType = 0
    # TODO: What's the meaning of the types of the sentences?
    # A.
    # meaning sing by which singer
    # meaning not the lyrics sentence (introduction sentences)
    # meaning the same singer singing with different vocal voice ! (ex.different pitch, different timbre, hibrid voice...)
    # meaning rap!!
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

    def print_info(self):
        print self.startTime,",",self.endTime,",",self.sentenceType,",",self.sentence

class LyricsInfo:
    ID = None
    sentenceInfos = []
    def __init__(self,array):
        self.ID = int(array[0])
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
    voc_dict = []
    def __init__(self,array):
        for element in array:
            li = LyricsInfo(element)
            self.lyricsinfos.append(li)
            self.ids.append(li.ID)
    # TODO: the function that can find an info of an corresponding lyrics
    def findInfobyID(self,id):
        return self.lyricsinfos[self.ids.index(id)]

    def dict_generate(self):
        if len(self.voc_dict)==0:
            voc_set = set()
            for element in self.lyricsinfos:
                voc_set = set().union(*[voc_set,element.voc_set()])
            voc_array = list(voc_set)
            voc_array.sort()
            self.voc_dict = (dict(zip(voc_array,range(len(voc_array)))),dict(zip(range(len(voc_array)),voc_array)))
            #string = ""
            #for lyrics in self.lyricsinfos:
            #    for sentenceinfo in lyrics.sentenceInfos:
            #        string = string.join(sentenceinfo.sentence)
            #        string.join(" ")

            #all_tokenized = string.split(' ')
            #voc_set = set(all_tokenized)
            #voc_array = list(voc_set)
            #voc_array.sort()
            #self.voc_dict = (dict(zip(voc_array,range(len(voc_array)))),dict(zip(range(len(voc_array)),voc_array)))
            return self.voc_dict
        else:
            return self.voc_dict
    def indexify(self,monitor=False):
        for lyrics in self.lyricsinfos:
            for sentence in lyrics.sentenceInfos:
                for i in range(len(sentence.tokenized_sentences)):
                    sentence.tokenized_sentences[i] = self.voc_dict[0][sentence.tokenized_sentences[i]]
            if monitor:
                print lyrics.ID


##### XXX MAIN CODE START FROM HERE !


#REVIEW: #### initialize the lyrics_data object from database
lyrics_data = LyricsData(song_lyrics_data)
song_info_data = SongInfoData(song_info)
# XXX form an voc list with voc id
voc_dict = lyrics_data.dict_generate()
lyrics_data.indexify()

#REVIEW:####loading the object#########################
fileh2 = open("lyricsdata",'r')
lyrics_data = pickle.load(fileh2)
fileh2.close()

#REVIEW:####saving the object###########################
import pickle
filehandler = open("lyricsdata",'w')
pickle.dump(lyrics_data,filehandler)
filehandler.close()
########################################################


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

#TESTING######################################################################

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
