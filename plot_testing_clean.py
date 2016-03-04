#TODO:
#XXX lyrics selection is wrong ! check the visualization.py
#XXX able to change size power, with DF or TF information remaining
#TODO label id to song, so that I can check with KKBOX radio player!
#TODO able to switch from TF to TFIDF
#TODO term frequency should be normalize by song word count before adding them together
#TODO summed term frequency should be devided by song number (normalize)
#TODO song selection by words


from bokeh.io import vform
from bokeh.plotting import figure, output_file, show, ColumnDataSource,hplot
from bokeh.models import HoverTool,CustomJS,Slider,BoxZoomTool, ResetTool,WheelZoomTool,PanTool,LassoSelectTool,Rect
import numpy as np
output_file("kk_c1_d64_walk_100_tsne_d2_cleaned(interactive)_id.html")
map_radius = 0.7
start_size = 5
start_alpha = 0.3
#REVIEW:input embedding object :
embedding = embedding_2D

lyrics_size = len(lyrics_data.ids)
voc_size = len(lyrics_data.voc_dict[0])


voc_list = [lyrics_data.voc_dict[1][i] for i in range(voc_size)]
len(voc_list)
voc_doc_freq = []
for element in voc_list:
    voc_doc_freq.append(lyrics_data.document_frequency[element])
#TESTING DOCUMENT FREQUENCY!
max(voc_doc_freq)
voc_doc_freq.index(max(voc_doc_freq))
voc_list[26963]
#import numpy as np
#import matplotlib.pylab as plt
#plt.hist(np.log(np.array(voc_doc_freq)/float(lyrics_size))+10,100)
#plt.show()
sorted_ids = np.sort(lyrics_data.ids)
voc_doc_freq_size = -np.log(np.array(voc_doc_freq)/float(lyrics_size))
title_list = [song_info_data.findInfobyID(element).title for element in sorted_ids]
album_list = [song_info_data.findInfobyID(element).album for element in sorted_ids]
artist_list = [song_info_data.findInfobyID(element).artist for element in sorted_ids]
#lyrics_list = [element.voc_dict() for element in lyrics_data.lyricsinfos]#TODO:here we should sort lyrics by ids
lyrics_list = [lyrics_data.lyricsinfos[lyrics_data.ids.index(e)].voc_dict() for e in sorted_ids]#lyrics_data.ids.index(e) for e in sorted_ids
#XXX extend the label list to full size

len(lyrics_list)
len(voc_list)
len(title_list)
len(album_list)
len(artist_list)
#REVIEW restart here!

source_song = ColumnDataSource(
        data=dict(
            x=embedding[voc_size:,0],
            y=embedding[voc_size:,1],
            size = [start_size]*(lyrics_size),
            title=title_list,
            album=album_list,
            artist=artist_list,
            ids = sorted_ids,
            #voc = voc_list,
            alpha=[start_alpha]*(lyrics_size),
            lyrics = lyrics_list
            #SorV = [0]*(voc_size)+[1]*(lyrics_size)
            )
    )
lyrics_data.lyricsinfos[0].voc_dict()
title_list[0]
album_list[0]
[voc_list[e] for e in lyrics_list[0].keys()]
#TODO change the source_voc to the displaying property, and elsewhere for storing DF TF
source_voc = ColumnDataSource(
        data=dict(
            x=embedding[:voc_size,0],
            y=embedding[:voc_size,1],
            size = voc_doc_freq_size.tolist(),#[start_size]*(voc_size),
            #title=title_list,
            #album=album_list,
            #artist=artist_list,
            voc = voc_list,
            alpha=[start_alpha]*(voc_size),
            color=["#%02x%02x%02x"%(255, e, 0) for e in voc_doc_freq_size.tolist()],
            #SorV = [1]*(voc_size)+[0]*(lyrics_size)
            docfreq = voc_doc_freq_size.tolist()

            )
    )

source_voc_fix = ColumnDataSource(
    data=dict(
        size = [start_size],
        alpha = [start_alpha]
    )
)
source_voc_size = ColumnDataSource(
    data=dict(
        size = voc_doc_freq_size.tolist()#[start_size]*(voc_size),
    )
)

##TODO create a selection callback
# voc color should be change according to voc count
# non selected words should be 0 alpha
# voc_count should be merge
# Two different mode:
# 1. after the selected words change the unselected words don't change ; (I can see different group clearly)
# 2. after the selected words change the unselected words recover.

source_song.callback = CustomJS(args=dict(source=source_voc,sf=source_voc_fix,svs=source_voc_size), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        var song_data = cb_obj.get('data');
        var voc_data = source.get('data');
        var voc_size = sf.get('data')['size'][0];
        var voc_alpha = sf.get('data')['alpha'][0];
        var voc_size_tmp = svs.get('data')['size'];
        function sum( obj ) {
            var sum = 0;
            for( var el in obj ) {
                if( obj.hasOwnProperty( el ) ) {
                  sum += parseFloat( obj[el] );
                }
            }
          return sum;
        }
        function normalize(voc){
            var summed = sum(voc);
            //console.log(Object.keys(voc));
            for(var el in voc){
                voc[el] = voc[el]/summed;
            }
            return voc;
        }
        Object.extend = function(destination, source) {
            for (var property in source) {
                if (destination.hasOwnProperty(property)) {
                    destination[property] += source[property];
                }else{
                    destination[property]=source[property];
                }
            }
            return destination;
        };
        var vocs = {

        };
        for (i = 0; i < inds.length; i++) {
            vocs = Object.extend(vocs, song_data['lyrics'][inds[i]]);
            //var key = Object.keys(song_data['lyrics'][inds[i]]);
            //for (j=0;j<key.length;j++){
            //    vocs.add(key[j]);
            //}
        }
        var voc_sum = sum(vocs);
        vocs = normalize(vocs);//do the normalization to all voc
        var key = Object.keys(vocs);
        var select_array = new Array(voc_data['alpha'].length);
        key.forEach(function(value) {
            select_array[value]=1;
        });
        function componentToHex(c) {
            var hex = c.toString(16);
            return hex.length == 1 ? "0" + hex : hex;
        }

        function rgbToHex(r, g, b) {
            return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
        }

        for (i=0;i<select_array.length;i++){
            if(inds.length==0){
                voc_data['alpha'][i] = voc_alpha;
                voc_data['size'][i] = voc_size_tmp[i];//TODO:Change to Document Freq.
                voc_data['color'][i] = rgbToHex(255, 0, 0);
            }else{
                if(select_array[i]==1){
                    voc_data['alpha'][i] = voc_alpha;
                    voc_data['size'][i] = Math.pow(vocs[i]*voc_data['docfreq'][i]*select_array.length/inds.length,voc_size);//TODO:Change to tf-idf with normalized voc tf with normalized by song count
                    //voc_data['size'][i] = Math.pow(vocs[i]*select_array.length/inds.length,voc_size);//TODO:Change to tf
                    //voc_data['size'][i] = Math.pow(vocs[i]*voc_data['docfreq'][i]*voc_sum,voc_size);//TODO:Change to tf-idf with normalized voc tf with normalized by term count
                    //voc_data['size'][i] = Math.pow(vocs[i]*voc_sum,voc_size);//TODO:Change to tf
                    voc_data['color'][i] = rgbToHex(255,voc_data['size'][i],0);//map from size
                    voc_size_tmp[i]=voc_data['size'][i];
                }else{
                    voc_data['alpha'][i] = 0;
                    voc_data['size'][i] = 0;
                    voc_data['color'][i] = rgbToHex(255, 0, 0);
                }
            }

        }

        source.trigger('change');
    """)
hover_song = HoverTool(
        tooltips=[
            ("title", "@title"),
            ("album", "@album"),
            ("artist", "@artist"),
            ("id", "@ids")
                    ]
    )
hover_voc = HoverTool(
        tooltips=[
            ("voc", "@voc")
        ]
    )
Tools_song = [BoxZoomTool(), ResetTool(),WheelZoomTool(),PanTool(),hover_song,LassoSelectTool()]
Tools_voc = [BoxZoomTool(), ResetTool(),WheelZoomTool(),PanTool(),hover_voc,LassoSelectTool()]
p_song = figure(tools=Tools_song,
           title="SONG",webgl=True)
p_voc = figure(tools=Tools_voc,
           title="VOC",webgl=True,x_range = p_song.x_range,y_range = p_song.y_range)
p_song.circle('x', 'y', size='size',fill_alpha = 'alpha',fill_color="#%02x%02x%02x"%(0, 255, 0),line_color="#%02x%02x%02x"%(0,255,0),line_alpha=0.8, source=source_song)
p_voc.circle('x', 'y', size='size',fill_alpha = 'alpha',fill_color='color',line_color=None, source=source_voc)


callback_alpha = CustomJS(args=dict(source_song=source_song,source_voc=source_voc,sf=source_voc_fix), code="""
        var song_data = source_song.get('data');
        var voc_data = source_voc.get('data');
        var f = cb_obj.get('value');
        //source_voc.get('alpha') = f;
        sf.get('data')['alpha'][0]=f;
        song_alpha = song_data['alpha'];
        voc_alpha = voc_data['alpha'];
        for (i=0;i<song_alpha.length;i++){
            song_alpha[i]=f;
        }
        for (i=0;i<voc_alpha.length;i++){
            voc_alpha[i]=f;
        }
        source_song.trigger('change');
        source_voc.trigger('change');
    """)
callback_size = CustomJS(args=dict(source_song=source_song,source_voc=source_voc,sf=source_voc_fix,svs=source_voc_size), code="""
        var song_data = source_song.get('data');
        var voc_data = source_voc.get('data');
        var f = cb_obj.get('value');
        var voc_size_tmp = svs.get('data')['size'];
        //source_voc.get('size') = f;
        sf.get('data')['size'][0]=f;
        song_size = song_data['size'];
        voc_size = voc_data['size'];
        for (i=0;i<song_size.length;i++){
            song_size[i]=f;
        }
        for (i=0;i<voc_size.length;i++){
            voc_size[i]=Math.pow(voc_size_tmp[i],f);
            //voc_size[i]=f;
        }
        source_song.trigger('change');
        source_voc.trigger('change');
    """)

##########################################################
slider_alpha = Slider(start=0.1, end=1, value=start_alpha, step=.1, title="fill_alpha", callback=callback_alpha)
slider_size = Slider(start=0.1, end=1, value=start_size, step=.1, title="size", callback=callback_size)

#construct the graph map for checking the location of zooming
colors_red = ["#%02x%02x%02x"%(int(r), int(g), 0) for r, g in zip([255]*voc_size, [0]*voc_size)]
colors_green = ["#%02x%02x%02x"%(int(r), int(g), 0) for r, g in zip([0]*lyrics_size, [255]*lyrics_size)]
colors = []
colors.extend(colors_red)
colors.extend(colors_green)


source = ColumnDataSource({'x': [], 'y': [], 'width': [], 'height': []})

jscode="""
        var data = source.get('data');
        var start = range.get('start');
        var end = range.get('end');
        data['%s'] = [start + (end - start) / 2];
        data['%s'] = [end - start];
        source.trigger('change');
    """
p_song.x_range.callback = CustomJS(
        args=dict(source=source, range=p_song.x_range), code=jscode % ('x', 'width'))
p_song.y_range.callback = CustomJS(
        args=dict(source=source, range=p_song.y_range), code=jscode % ('y', 'height'))
p_map = figure(title='See Zoom Window Here',
            tools='', plot_width=300, plot_height=300,webgl=True)

p_map.scatter(x=embedding[:,0], y=embedding[:,1], radius=map_radius, fill_alpha=0.08,color=colors, line_color=None)

rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.3,
            line_color='black', fill_color='black')
p_map.add_glyph(source, rect)

show(vform(hplot(p_map,vform(slider_alpha,slider_size)),hplot(p_voc,p_song)))
