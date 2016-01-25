

# XXX read data from file

artists = [line.rstrip('\n') for line in open('artists')]
titles = [line.rstrip('\n') for line in open('title')]


lyrics = []
for i in range(len(titles)):
    lyrics.append(open('lyrics/' + artists[i] + '-' + titles[i]).read())

len(lyrics)
lyrics

# TODO try to use  PunktWordTokenizer to tokenize lyrics
from nltk.tokenize.punkt import PunktToken
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
sb = SnowballStemmer('english')
lc = LancasterStemmer()


# XXX richness analysis
import matplotlib.pyplot as plt
richness = []
for element in lyrics:
    richness.append(len(set(element.split())) / len(element.split()))

len(set(element.split()))
len(element.split())


plt.plot(sorted(richness, reverse=True))
plt.show()

# with snowball stemming
richness = []
for element in lyrics:
    richness.append(len(set([sb.stem(w) for w in element.split()])
                        ) / len([sb.stem(w) for w in element.split()]))

plt.plot(sorted(richness, reverse=True))
plt.show()

# with Lancaster Stemming
richness = []
for element in lyrics:
    richness.append(len(set([lc.stem(w) for w in element.split()])
                        ) / len([lc.stem(w) for w in element.split()]))

plt.plot(sorted(richness, reverse=True))
plt.show()


# XXX richness of example books in nltk
text_richness = []

from nltk.corpus import gutenberg
for fileid in gutenberg.fileids():
    tokens = gutenberg.words(fileid)
    text_richness.append(len(set(tokens)) / len(tokens))
plt.plot(sorted(text_richness, reverse=True))
plt.show()

print(gutenberg.raw(gutenberg.fileids()[3]))

# XXX compare the similarity of each text
# generate the total vocaburay of all lyrics

from nltk import *
lyrics_token = []

for lyric in lyrics:
    lyrics_token.append([lc.stem(w) for w in word_tokenize(lyric)])

lyrics_token_append = []
for lyric_token in lyrics_token:
    lyrics_token_append.extend(lyric_token)
len(lyrics_token_append)
voc_count = len(set(lyrics_token_append))
voc_count
voc_table = FreqDist(lyrics_token_append).most_common(voc_count)
voc_table


voc_table[0][0]

# XXX generate vector for each lyrics
from numpy import *

lyric_vecs = zeros((voc_count, len(lyrics)))
# shape(vec_test)

for j in range(len(lyrics)):
    for i in range(voc_count):
        lyric_vecs[i][j] = lyrics_token[j].count(voc_table[i][0])


plt.imshow(lyric_vecs)
plt.show()

shape(lyric_vecs)

lyrics_matrix = matrix(lyric_vecs)

# XXX generate similarity matrix version 1

lyrics_matrix
similarity_matrix = lyrics_matrix.T * lyrics_matrix

plt.imshow(similarity_matrix)
shape(similarity_matrix)
plt.show()

# XXX plot histogram of similarity matrix
norm_similarity_matrix = similarity_matrix / \
    max(max(similarity_matrix.tolist()))
max(max(norm_similarity_matrix.tolist()))

all_similarity = []

for line in norm_similarity_matrix.tolist():
    all_similarity.extend(line)

plt.hist(all_similarity, bins=5000)
plt.show()

# XXX generate similarity matrix version 2 (distance matrix)
from scipy import sparse
sparse_matrix = sparse.csr_matrix(lyric_vecs)
sparse_matrix

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

distance_matrix = pairwise_distances(sparse_matrix.T, metric="cosine")
shape(distance_matrix)

plt.imshow(distance_matrix)
plt.show()

# XXX generate histogram of distance matrix version 2
distance_matrix
all_distance = []
for line in distance_matrix:
    all_distance.extend(line)
plt.hist(all_distance, bins=5000)
plt.show()

# XXX tf-idf VSM
# get rid of stop word
# calculate term frequency vector  (tf) :
# calculate inverse document frequency (idf):
# idf(t) = log(|Document count|/(1+{the number of documents that include t)})

# TODO here I can try to make previous simple cos distance algorithm
# supported by sklearn.feature_extraction.text in order to make it clearer
from sklearn.feature_extraction.text import CountVectorizer


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, lc)
    return stems
count_vectorizer = CountVectorizer(tokenizer=tokenize)
count_vectorizer.fit_transform(lyrics)
count_vectorizer.vocabulary_
feature_name = count_vectorizer.get_feature_names()
freq_term_matrix = count_vectorizer.transform(lyrics)
plt.plot(freq_term_matrix.todense()[0].T)
plt.show()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tfidf.idf_
tf_idf_matrix = tfidf.transform(freq_term_matrix)
tf_idf_matrix.todense

distance_matrix_tfidf = pairwise_distances(tf_idf_matrix, metric="cosine")
shape(distance_matrix_tfidf)

plt.imshow(distance_matrix_tfidf)
plt.show()

# XXX plot the histogram of distance (similarity) to see the distribution
# of similarity
distance_matrix_tfidf
all_distance_tfidf = []
for line in distance_matrix_tfidf:
    all_distance_tfidf.extend(line)
plt.hist(all_distance_tfidf, bins=5000)
plt.show()
# seems most lyrics are very unsimilar , only 5~10 lyrics are similar !

# XXX try to compare the relation of common distance matrix and tf-idf
# distance matrix
shape(all_distance)
shape(all_distance_tfidf)
plt.plot(all_distance, all_distance_tfidf, 'ro')
plt.show()

# XXX find the closest lyrics to lyric n, instead of itself


def similarLyricsRank(lyric_index, k):
    """ this function help find the k'th similar lyrics index given an lyrics by index
    @lyric_index:the index of lyrics we wish to find the similar lyrics accordingly
    @k:
    @return:the k'th similar lyrics
    """
    import heapq
    return list(distance_matrix[lyric_index]).index(heapq.nsmallest(k, distance_matrix[lyric_index])[k - 1])

# XXX checking
similarLyricsRank(0, 2)
artists[0], titles[0]
artists[78], titles[78]
print(lyrics[3])
print(lyrics[107])

#  XXX find two most similar lyrics:
similarest_list = []
for index in range(len(lyrics)):
    similarest_list.append(
        similarity_matrix[index, similarLyricsRank(index, 2)])

# XXX checking
titles
similarest_list
argmin(similarest_list)
similarLyricsRank(120, 2)


artists[120], titles[120]
artists[230], titles[230]
print(lyrics[120])
print(lyrics[230])

# some special case : 38/129 , 207/232

# XXX generate word cloud :
# paste the output of the result to "http://www.wordle.net/advanced"

lyricsindex = 129
for index in range(len(feature_name)):
    rate = tf_idf_matrix.todense()[lyricsindex, index]
    if rate != 0.:
        print(feature_name[index], ":", rate)
