
# XXX read data from file

artists = [line.rstrip('\n') for line in open('artists')]
titles = [line.rstrip('\n') for line in open('title')]

lyrics = []
for i in range(len(titles)):
    lyrics.append(open('lyrics/' + artists[i] + '-' + titles[i]).read())

len(lyrics)
lyrics

# XXX tokenize all lyrics
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
