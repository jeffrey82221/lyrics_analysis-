from __future__ import print_function
import sys
try:
    cf_filename = sys.argv[1]
    dimension = sys.argv[2]
except:
    print("no enought argument input ! \n should input :\n 1.CF file name 2. dimension")
    exit()

print('cf_filename = ',cf_filename)
filename = 'win1.w100.l32.d128.embeddings.2d'
import numpy as np
import pickle
import numpy as np
import pickle
voc_dict=pickle.load(open('dict.voc','rb'))
voc_size = len(voc_dict[0])
keys = np.loadtxt(filename+'.keys').astype(int)
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

#TODO match song with its cf embeddings

song_cffile =  open(cf_filename, 'rb')
song_em=np.loadtxt(song_cffile)
song_index2order=dict(zip(song_em[:,0].astype(int).tolist(),song_em[:,1].astype(int).tolist()))



order_keys = []
no_match_keys = []
no_matched_count = 0
for e in song_keys:
    try:
        order_keys.append(song_index2order[e])
    except:
        no_matched_count = no_matched_count+1
        order_keys.append(-1)
print('no matched count : ',no_matched_count)
len(song_keys)
len(np.array(order_keys)==-1)
song_em_matched=song_em[order_keys,2:]
song_em_matched[np.array(order_keys)==-1,:]=np.zeros(np.shape(song_em_matched[np.array(order_keys)==-1,:]))
embedding_matrix=np.matrix(song_em_matched)
#dimension reduction
print('matched embedding shape = ',np.shape(song_em_matched))

import sklearn
def cosine_distance(X1,X2):
    return sklearn.metrics.pairwise.pairwise_distances(X1,X2, metric='cosine')

try:
    from tsne import bh_sne
    print("start transforming...")
    embedding_low = bh_sne(embedding_matrix,d=int(dimension),theta=0.2)
except:
    from sklearn.manifold import TSNE
    print("start transforming...")

    model = TSNE(n_components=int(dimension), perplexity=30.0, early_exaggeration=10.0, learning_rate=1000.0,  n_iter=1000, metric='euclidean', init='pca',angle=0.2)
    embedding_low = model.fit_transform(embedding_matrix)

print('result shape = ',np.shape(embedding_low))

from pandas import DataFrame


song_dict = dict()
song_dict['title']=title_list
song_dict['album']=album_list
song_dict['artist']=artist_list

for i in range(dim):
    song_dict['d'+str(i)]=embedding_low[:,i]

song_df=DataFrame(song_dict)

print(song_df)
song_df.to_excel(cf_filename+'.'+dimension+'d'+'.xlsx', sheet_name='sheet1', index=False)
