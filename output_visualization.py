import matplotlib.pyplot as plt
%matplot inline
# XXX readin embedding of each object
result_lines = [line.rstrip('\n') for line in open('out2.embeddings')]
result_lines
# deepwalk (skipgram) info : (object number , representation dimension)
result_lines[0]

object_count = len(result_lines)
splited_result_lines = []
for i in range(1, len(result_lines)):
    splited_result_lines.append(result_lines[i].split())

len(splited_result_lines)
embedding_list = []
for items in splited_result_lines:
    embedding_list.append(
        (int(items[0]), [float(item) for item in items[1:-1]]))


embedding_list.sort()

embedding = dict(embedding_list)

from numpy import *
# change embedding data into matrix
embedding_list = []
for i in range(1, len(embedding) + 1):
    embedding_list.append(embedding[i])
embedding_matrix = matrix(embedding_list)



# TODO calculate an distance matrix of each object : use ecludien distance
from sklearn.metrics import pairwise_distances
distance_matrix = pairwise_distances(embedding_matrix, metric="euclidean")
shape(distance_matrix[0:266, 0:266])
shape(distance_matrix[266:, 266:])
plt.imshow(distance_matrix[0:266, 0:266])
plt.show()
# split to different kind of matrix
song_distance_matrix = distance_matrix[0:266, 0:266]
song_word_distance_matrix = distance_matrix[0:266, 266:]
word_distance_matrix = distance_matrix[266:, 266:]

# plot the histogram of distance matrix
def matrix_hist(matrix):
    matrix_value = []
    for line in matrix:
        matrix_value.extend(line)
    plt.hist(matrix_value, bins=5000)
    plt.show()

matrix_hist(song_distance_matrix)
matrix_hist(word_distance_matrix)
matrix_hist(song_word_distance_matrix)

def similarRank(distance_matrix, index, k):
    """ this function help find the k'th similar object index given an index
    @index:the index of object we wish to find the similar object accordingly
    @k:
    @return:the k'th similar object
    """
    import heapq
    return list(distance_matrix[index]).index(heapq.nsmallest(k, distance_matrix[index])[k - 1])

# XXX find the similarest word or song
similarRank(song_distance_matrix, 120, 2)
similarRank(song_word_distance_matrix, 3, 2)
similarRank(word_distance_matrix, 3, 2)

# TODO readin the represented object of the index

# read in the info of the songs
artists = [line.rstrip('\n') for line in open('artists')]
titles = [line.rstrip('\n') for line in open('title')]
lyrics = []
for i in range(len(titles)):
    lyrics.append(open('lyrics/' + artists[i] + '-' + titles[i]).read())

# read in the voc
voc = [line.rstrip('\n') for line in open('voc')]
voc.sort()

# XXX find the most similar object from song to voc
voc[120]
voc[3164]

titles[120]
titles[230]
artists[120]
artists[230]
print(lyrics[120])
print(lyrics[230])

# XXX T-SME visualization of result
from sklearn.manifold import TSNE
model = TSNE(n_components=2, perplexity=10.0, early_exaggeration=10.0, learning_rate=1000.0,
     n_iter=1000, metric='euclidean', init='random')
embedding_3D = model.fit_transform(embedding_matrix)

# TODO generate toks, vs_sne, cats, tok_cats for web interative visualization

# XXX toks
toks = []
for i in range(len(voc)):
    toks.append(voc[i])
for i in range(len(titles)):
    toks.append(titles[i])
len(toks)
# XXX vs_sne
vs_sne = []
for i in range(len(embedding_3D)):
    vs_sne.append(embedding_3D[i][0])
    vs_sne.append(-embedding_3D[i][1])
len(vs_sne)
# exchange the order of song and voc
vs_sne_e = vs_sne[len(titles)*2:]
vs_sne_e.extend(vs_sne[:len(titles)*2]) #the titles

len(vs_sne_e)
vs_sne = vs_sne_e

# XXX cats & tok_cats
# NOTE list out the sequence of the type of category as reference!!
# (SONG:    song object)
# (VOC:     vocabulary object)
# (ARTIST.  song object with the artist)
# (WITHTITLE. vocabulary object within the lyrics of title's song)
# (WITHVOC.    song object with the vocabulary)
# (WITHARTIST. vocabulary object within the lyrics of the artist's song)
# TODO adding the exact song or voc finding tag

# cats of SONG & VOC
cats = []
cats.append("SONG")
cats.append("VOC")
artist_set = list(set(artists))

# cats of ARTIST
for i in range(len(artist_set)):
    cats.append("ARTIST."+artist_set[i])

# cats of WITHTITLE
for t in titles:
    cats.append("WITHTITLE."+t)
# cats of WITHTVOC
for v in voc:
    cats.append("WITHVOC."+v)
# cats of WITHARTIST
for a in artist_set:
    cats.append("WITHARTIST."+a)

# XXX tok_cats
# reading the category of words & song from the adjlistfile
adjlistfile = open('in.adjlist')
adjlist = [[int(k) for k in element.split()] for element in adjlistfile]
len(adjlist)
tok_cats = []
# tok_cats of VOC and SONG
for i in range(len(voc)):
    tok_cats.append([1])
for i in range(len(titles)):
    tok_cats.append([0])
# tok_cats of ARTIST
for i in range(len(titles)):
    tok_cats[3999+i].append(cats.index("ARTIST."+artists[i]))


# tok_cats of WITHTITLE
# the index of titles should be the index in adjlist -1
# add the titles cat to the voc toks
for i in range(len(voc)):
    for j in range(len(adjlist[266+i])-1):
        tok_cats[i].append(cats.index("WITHTITLE."+titles[adjlist[266+i][j+1]-1]))

# tok_cats of WITHVOC
cats.index("WITHVOC."+voc[adjlist[i][j+1]-1-266])
tok_cats[3999]
for i in range(len(titles)):
    for j in range(len(adjlist[i])-1):
        tok_cats[i+3999].append(cats.index("WITHVOC."+voc[adjlist[i][j+1]-1-266]))

# tok_cats of WITHARTIST
for i in range(len(voc)):
    for j in range(len(adjlist[266+i])-1):
        tok_cats[i].append(cats.index("WITHARTIST."+artists[adjlist[266+i][j+1]-1]))


# XXX print them out in the disried formate!
# print the toks
import sys
for i in range(1):
    for j in range(50000):
        sys.stdout.write("'")
        if j<len(toks):
            sys.stdout.write(string.replace(string.replace(toks[j], ' ', '-', 10),"'","",10))
        else:
            sys.stdout.write("")
        sys.stdout.write("'")
        sys.stdout.write(",")
    sys.stdout.write("\n,\n")

# print the vs_sne
for i in range(1):
    for j in range(100000):
        if j<len(vs_sne):
            sys.stdout.write(str(vs_sne[j]))
        else:
            sys.stdout.write(str(0.0))
        sys.stdout.write(",")
    sys.stdout.write("\n,\n")

# print the cats
import string
for i in range(1):
    for element in cats:
        sys.stdout.write("'")
        sys.stdout.write(string.replace(string.replace(element, ' ', '-', 10),"'","",10))
        sys.stdout.write("'")
        sys.stdout.write(",")
    sys.stdout.write("\n,\n")
# print the tok_cats
for i in range(1):
    for k in range(50000):
        sys.stdout.write("[")
        if k<len(tok_cats):
            for j in range(len(tok_cats[k])):
                sys.stdout.write(str(tok_cats[k][j]))
                if j<len(tok_cats[k])-1:
                    sys.stdout.write(",")
            else:
                sys.stdout.write("")
        sys.stdout.write("]")

        sys.stdout.write(",")
    sys.stdout.write("\n,\n")



# plot the dimension reduced embedding in pdf file
# XXX color the result with different category
c_list = []
s_list = []
scale = 20
for i in range(266+3999):
    if(i<266):
        c_list.append('r')
        s_list.append(400*scale)
    else:
        c_list.append('g')
        s_list.append(50*scale)
#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10*scale,10*scale))
ax = fig.add_subplot(111)
ax.grid(True,linestyle='-',color='0.75')
ax.scatter(array(embedding_3D[:, 0].T), array(
    embedding_3D[:, 1].T),c=c_list[:],s=s_list[:],alpha=0.4,edgecolor='none')

len(voc)
#len(titles)
len(titles)
#XXX avoid txt overlapping :
#REF http://stackoverflow.com/questions/8850142/matplotlib-overlapping-annotations

def get_text_positions(x_data, y_data, txt_width, txt_height):
    import numpy as np
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions
def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height,txts):
    import numpy as np
    for x,y,t,txt in zip(x_data, y_data, text_positions,txts):
        try: #TODO the unicode convertion is up to solve
            axis.text(x, 1.0*t, unicode(txt),rotation=0, color='black')
            if y != t:
                axis.arrow(x, t,0,y-t, color='blue',alpha=0.3, width=txt_width*0.005,
                           head_width=txt_width*0.1, head_length=txt_height*0.5,
                           zorder=0,length_includes_head=True)
        except:
            print txt

txt_height = 0.05
txt_width = 0.1
text_positions = get_text_positions(array(embedding_3D[266:, 0].T), array(
    embedding_3D[266:, 1].T), txt_width, txt_height)
text_plotter(array(embedding_3D[266:, 0].T), array(
    embedding_3D[266:, 1].T), text_positions, ax, txt_width, txt_height,voc)


for i in range(len(titles)):
    try: #TODO the unicode convertion is up to solve
        ax.annotate(unicode(titles[i]), (embedding_3D[i,0],embedding_3D[i,1])
            ,color="brown",size=24)
    except:
        print i
fig.savefig('new_out2.pdf')

plt.show()



# TODO further anlaysis of result : COLOR TF OR DF , CHECK THE OUTSIDER !

# anlaysis of word_distance_matrix, what are the clustered part ?
shape(word_distance_matrix)
check_indexes_in = argwhere(bitwise_and(word_distance_matrix<1,word_distance_matrix>0))
check_indexes_in

voc_pair_in = []
for i in range(len(check_indexes_in)):
    voc_pair_in.append([voc[check_indexes_in[i][0]],voc[check_indexes_in[i][1]]])


# anlaysis of word_distance_matrix, what are the out of clustered part ?
shape(word_distance_matrix)
check_indexes_out = argwhere(word_distance_matrix>=1)

voc_pair_out = []
for i in range(len(check_indexes_out)):
    voc_pair_out.append([voc[check_indexes_out[i][0]],voc[check_indexes_out[i][1]]])

# print them out
for pair in voc_pair_in:
    print pair[0],pair[1]


# print them out
for pair in voc_pair_out:
    print pair[0],pair[1]


# find the voc in the closer group :
voc_in_list = []
[voc_in_list.extend(element) for element in voc_pair_in]
voc_out_list = []
[voc_out_list.extend(element) for element in voc_pair_out]

voc_in_set = set(voc_in_list)
voc_out_set = set(voc_out_list)

voc_out_set = voc_out_set.symmetric_difference(voc_in_set)
voc_in_set
voc_out_set

# TODO find the document frequency of each words in this two set, since we assume
# that the song can ties the words up , if the words are close, means it maybe appear in many songs, if the words are more outlie,
# then means it maybe appear last in many songs.



# TODO use 3D or 2D deepwalk SOLELY, instead of using TSME
