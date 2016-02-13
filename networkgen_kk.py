



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

####TESTING the cleaning process





sb = nltk.SnowballStemmer("english", ignore_stopwords=True)
wnl = nltk.WordNetLemmatizer()
tagger = nltk.tag.PerceptronTagger()
stemming_function_array = [remove_front_mark,remove_end_mark,sb.stem,wnl.lemmatize]
len(set([applystemming(t,[remove_front_mark,remove_end_mark]) for t in voc_list]))

#TODO:I can use the edition distance as an idication for stemming the same vocabulary (voc in lyrics often repeat or contain some differences!)
sb.stem('gone')
wnl.lemmatize('gone', pos='v')

from nltk.corpus import wordnet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''




#NOTE:pos_tag before stem to get a more accurate lemmatize reference
pos_tag = nltk.pos_tag(nltk.word_tokenize("I studied late. "))



pos_tag = nltk.tag._pos_tag(['I','am','testing','the','tagger'], None, tagger)


pos_tag_lem = [(element[0],get_wordnet_pos(element[1])) for element in pos_tag]

[wnl.lemmatize(t[0],pos=t[1]) if t[1]!='' else t[0] for t in pos_tag_lem]



nltk.pos_tag([sb.stem(t) for t in nltk.word_tokenize("I study late.")])
wnl.lemmatize("is",pos='v')
remove_front_mark(remove_end_mark(''))

tokens = SentenceInfo(song_lyrics_data[0][1][1]).tokenized_sentences
tokens

word_tokenize(tokens)
pos_tag(word_tokenize("doesn't"))

original_list = list(voc_list)
cleaned_list = list([sb.stem(t) for t in voc_list])

#REVIEW:testing the difference of two stemmed voc
lem_list = np.array(cleaned_list)[np.array(cleaned_list)!=np.array(original_list)].tolist()
ori_list = np.array(original_list)[np.array(cleaned_list)!=np.array(original_list)].tolist()


for p,l in zip(lem_list,ori_list):
    print p,l









wnl.lemmatize('doctors')
len(set(cleaned_voc_list))
len(set(stemmed_voc_list))
len(set(lemmatized_voc_list))
set(lemmatized_voc_list)




##### XXX MAIN CODE START FROM HERE !
# XXX read data from file ##########################################################
from detecting_language import *
import numpy as np
song_info = [element.split('\t') for element in [line.rstrip('\n') for line in open('data/Western_songs_info.tsv')]]
song_ids = [element for element in [line.split(',') for line in open('data/Western_songs.csv')][0]]
song_lyrics_tmp = [element.split('\t') for element in open('data/Western_songs_lyrics.tsv')]
song_lyrics_data = [[element[0],element[1].split('\\n')] for element in song_lyrics_tmp]
song_info
song_lyrics_data[0][1]
#TODO:remove none english songs :
#detecting language
language_tag = [get_language(''.join([sen for sen in element[1]])) for element in song_lyrics_data]

new_song_lyrics_data = [song_lyrics_data[i]  for i in range(len(language_tag)) if language_tag[i]=='english']
len(new_song_lyrics_data)

##################################################################################

#REVIEW:Language Analysis :
#NOTE:
# 1. Chinise songs are mistakenly detected
# 2. Song with no vocal are mistakenly detected
# 3. Possiblly there are some songs are of special spelling language, for example, marlesian ,indonisian, indian,etc...
set(language_tag)
language_tag.count('english')
#TODO:to much to check
language_tag.count('danish')
#NOTE:only the first song are possible danish song
language_tag.count('french')
#NOTE:English Songs with a lot of 'la' are mistakenly detected as french songs
language_tag.count('german')
#NOTE:the first 4 songs are all germen, but the last one are not.
language_tag.count('hungarian')
#NOTE:all hungarian detected songs are actually chinese songs
language_tag.count('italian')
#NOTE:
language_tag.count('norwegian')
#NOTE:the second one is chinese
language_tag.count('portuguese')
#NOTE:one is chinese and one is korean
language_tag.count('spanish')
#NOTE:one are chinese
language_tag.count('swedish') #Strange
#NOTE:most are in Chinese , Japanese , Korean

def findAllSongsOfLanguage(song_info_data,song_lyrics_data,language_tag,language):
    #result = []
    for i in range(len(language_tag)):
        if(language_tag[i]==language):
            t = song_info_data.findInfobyID(int(song_lyrics_data[i][0]))
            print t.title,t.album,t.artist
            for element in song_lyrics_data[i][1]:
                print element

    #return result
findAllSongsOfLanguage(song_info_data,song_lyrics_data,language_tag,'spanish')

####################################################################################
# import class that can fatch the lyrics and song data
import nltk
from ReadInfo import SongInfo,SongInfoData,SentenceInfo,LyricsInfo,LyricsData
#REVIEW: #### initialize the lyrics_data object from database
#TODO:tokenization is still not very accurate!!
#TODO:remove the sentence without words
lyrics_data = LyricsData(new_song_lyrics_data)
lyrics_data.lyricsinfos[99].sentenceInfos[4].tokenized_sentences
lyrics_data.lyricsinfos[99].sentenceInfos[4].pos_tags
lyrics_data.lyricsinfos[216].print_lyrics()
lyrics_data.lyricsinfos[216].print_info()
len(lyrics_data.lyricsinfos[0].sentenceInfos)
song_info_data = SongInfoData(song_info)
# XXX form an voc list with voc id
voc_dict = lyrics_data.dict_generate()
lyrics_data.indexify()
####################################################################################
voc_list = voc_dict[0].keys()
voc_list


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





###################################################################################
# TODO create an adj matrix of song id and voc id

# 14883 lyrics data provided !
# lyrics_data_ids = [int(element) for element in lyrics_data.ids] #574723~63476142

len(lyrics_data.ids)
len(song_info_data.ids)
lyrics_data.lyricsinfos[0].voc_set()
voc_dict[0].keys()
lyrics_data.lyricsinfos[1].voc_set()
adjlist = []
for element in lyrics_data.lyricsinfos:
    adjlist.extend(zip([element.ID]*len(element.voc_set()),list(element.voc_set())))

# generate adjlist file

adjlist
outfile = open("inkk_cleaned.adjlist",'w')
for element in adjlist:
    outfile.write(str(element[0]))
    outfile.write(" ")
    outfile.write(str(element[1]))
    outfile.write("\n")

outfile.close()
###################################################################################
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
