
from __future__ import print_function

import sys

filename = 'win7.w100.l32.d128.embeddings'
cf_filename = 'song-embedding-128.csv'
#TODO: input song-voc embedding (*.embeddings)

import numpy as np
result_lines = [line.rstrip('\n') for line in open(filename)]
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
embedding_all = np.matrix(embedding_array)
#TODO: adding voc.dict for voc size for seperation of song embedding and voc embedding
import pickle
voc_dict=pickle.load(open('dict.voc','rb'))
voc_size = len(voc_dict[0])
song_lyrics_em = embedding_all[voc_size:,:]
song_keys = embedding_key[voc_size:]



#TODO: input song cf embeddings (*.csv)

song_cffile = open('CF data/' + cf_filename, 'rb')
song_cf_em = np.loadtxt(song_cffile)
song_index2order = dict(zip(song_cf_em[:, 0].astype(
    int).tolist(), song_cf_em[:, 1].astype(int).tolist()))


#TODO: select the matched song embedding out of the two embedding

order_keys = []
match_keys = []
no_matched_count = 0
for e in song_keys:
    try:
        order_keys.append(song_index2order[e])
        match_keys.append(1)
    except:
        no_matched_count = no_matched_count + 1
        order_keys.append(-1)
        match_keys.append(0)


print('no matched count : ', len(match_keys)-sum(match_keys))

song_lyrics_em_matched = song_lyrics_em[np.array(match_keys)==1,:]

song_cf_em_matched = song_cf_em[order_keys, 2:]
song_cf_em_matched_ = np.matrix(song_cf_em_matched[np.array(match_keys)==1,:])
print("matched_count = ",sum(match_keys))
print("song_lyrics_em = ",np.shape(song_lyrics_em_matched))
print("song_cf_em = ",np.shape(song_cf_em_matched_))

#TODO: seperate the song embedding into training set ,validation set and testing set
input_data = song_lyrics_em_matched
output_data = song_cf_em_matched_
(total_size,dim)=np.shape(output_data)
print('normalization of each dimension...')

for i in range(np.shape(input_data)[1]):
    input_data[:,i]=(song_lyrics_em_matched[:,i]-np.mean(song_lyrics_em_matched[:,i]))/(np.var(song_lyrics_em_matched[:,i])**0.5)
    output_data[:,i]=np.transpose(np.matrix(np.random.randn(total_size)))


print('seperate data into train,validation and test set...')
train_size = 12000
val_size = 1200
test_size = np.shape(output_data)[0]-train_size-val_size

train_data = (input_data[:train_size,:],output_data[:train_size,:])
val_data = (input_data[train_size:train_size+val_size,:],output_data[train_size:train_size+val_size,:])
test_data = (input_data[train_size+val_size:,:],output_data[train_size+val_size:,:])

print('constructing the network')
#TODO: construct a neural network with song-voc embedding as input and song cf embedding as output
import tensorflow as tf
def weight_variable(shape,name = "W"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name = name)
def bias_variable(shape,name = "b"):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name = name)

X = tf.placeholder("float", shape=[None, 128])
Y = tf.placeholder("float", shape=[None, 128])
W1 = weight_variable([128, 128],name = "W1")
B1 = bias_variable([128],name = "B1")
W2 = weight_variable([128, 128],name = "W2")
B2 = bias_variable([128],name = "B2")
W3 = weight_variable([128, 128],name = "W3")
B3 = bias_variable([128],name = "B3")

with tf.device('/gpu:0'):
    H1 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(X, W1) + B1), 1.)
    H2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(H1, W2) + B2), 1.)
    O = tf.matmul(H2, W3) + B3
    square = tf.square(Y-O)
with tf.device('/cpu:0'):
    loss = tf.reduce_mean(square)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#TODO: create a function for generating batch in every training iteration
def batch(i,length,data):
    assert np.shape(data[0])[0]%length==0
    i = i%(np.shape(data[0])[0]/length)
    global perm
    if i ==0:
        perm = np.random.permutation(np.shape(data[0])[0])
    return (data[0][perm[i*length:(i+1)*length],:],data[1][perm[i*length:(i+1)*length],:])

print('start optimizing...')
for i in range(100000):
    (x,y)=batch(i,12000,train_data)
    if i%100==0:
        print(sess.run(loss,feed_dict={X:val_data[1], Y:val_data[0]}))
    sess.run(train,feed_dict={X:y, Y:x})
