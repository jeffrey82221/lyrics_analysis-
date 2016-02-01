
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
len(song_info_data.ids)

#REVIEW:####load the reduction 2D embedding###########################
import pickle
pfile = open("outkk_c1_d64_walk_10_tsne_d2",'r')
embedding_2D = pickle.load(pfile)
pfile.close()

#XXX Load the un reduce-dimensionalized embedding
result_lines = [line.rstrip('\n') for line in open('kk_c1_d2_walk_100.embeddings')]
len(result_lines)

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

#XXX check if two embedding matrix are in same data size
np.shape(embedding_2D)
np.shape(embedding_matrix)



# TODO : visualize the scattering using Bokeh

#TODO:number of voc as color bound
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool,Slider,CustomJS,ColumnDataSource

lyrics_size = len(lyrics_data.ids)
voc_size = len(lyrics_data.voc_dict[0])
lyrics_size
voc_size

colors_red = ["#%02x%02x%02x"%(int(r), int(g), 0) for r, g in zip([255]*voc_size, [0]*voc_size)]
colors_green = ["#%02x%02x%02x"%(int(r), int(g), 0) for r, g in zip([0]*lyrics_size, [255]*lyrics_size)]
colors = []
colors.extend(colors_red)
colors.extend(colors_green)

Tools = ['box_zoom','crosshair','resize','reset','pan','wheel_zoom']
#TODO : Adding more better tools !
hover = HoverTool(
            tooltips=[
                ("Voc", "@voc"),
                ("Title", "@title"),
                ("Artist", "@artist"),
                ("Album", "@album")
            ]
        )

data_dict = {
        'voc':player_data['Player'],
        'title':player_data['status'],
        'artist':player_data['ADP'],
        'album':player_data['Avg'],
    }
source_1 = ColumnDataSource(data=data_dict)
Tools.append(hover)


source = ColumnDataSource(data = dict(fill_alpha=0.1))

p = figure(tools=Tools,webgl=True)


embedding_matrix = np.array(embedding_matrix.tolist())

p.scatter(embedding_2D[:,0],
          embedding_2D[:,1],
          fill_color=colors,
          line_color=None,
          fill_alpha=0.1)


## setting up slider constrol ###########
callback = CustomJS(args=dict(source=p), code="""
        var data = source.get('fill_alpha');
        var alpha = cb_obj.get('value');
        data = alpha;
        source.trigger('change');
    """)
##########################################################


slider = Slider(start=0.1, end=1, value=1, step=.1, title="fill_alpha", callback=callback)
output_file("kk_c1_d64_walk_10_tsne_d2_new.html", title="kk_c1_d64_walk_10_tsne_d2_new")
show(p)


# XXX cats & tok_cats
# TODO: adding different color the different category
# TODO: fatch the category belonging from data info class
# NOTE list out the sequence of the type of category as reference!!
# (SONG:    song object)
# (VOC:     vocabulary object)
# (ARTIST.  song object with the artist)
# (WITHTITLE. vocabulary object within the lyrics of title's song)
# (WITHVOC.    song object with the vocabulary)
# (WITHARTIST. vocabulary object within the lyrics of the artist's song)


# TODO further anlaysis of result : COLOR TF OR DF , CHECK THE OUTSIDER !

# anlaysis of word_distance_matrix, what are the in clustered part ?

# anlaysis of word_distance_matrix, what are the out of clustered part ?


# find the voc in the closer group :


# TODO find the document frequency as radius of the dot of each words in this two set, since we assume
# that the song can ties the words up , if the words are close, means it maybe appear in many songs, if the words are more outlie,
# then means it maybe appear last in many songs.
# TODO find how the words is keywords as the radious of the dot of each song
# Which means that the sum of the tf-idf as the radius of the dot of each song
# TODO use 3D or 2D deepwalk SOLELY, instead of using TSME (Create another program)
