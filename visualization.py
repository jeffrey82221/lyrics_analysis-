from __future__ import print_function
try:
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
except:
    print("no enought argument input ! \n should input :\n 1. input filename 2. output filename")
    exit()

#REVIEW:####load the reduction 2D embedding###########################
import numpy as np
import pickle
voc_dict=pickle.load(open('dict.voc','rb'))
keys = np.loadtxt(inputfilename+'.keys').astype(int)
voc_size = len(voc_dict[0])
voc_keys = keys[:voc_size]
song_keys = keys[voc_size:]
lyrics_size = len(song_keys)
em = np.loadtxt(inputfilename)
embedding = np.matrix(em)

#TODO input song title, artist, album for each song
song_info = [element.split('\t') for element in [line.rstrip('\n')
                                                 for line in open('data/Western_songs_info.tsv')]]

from ReadInfo import *
song_info_data = SongInfoData(song_info)
artist_list = [song_info_data.findInfobyID(sk).artist for sk in song_keys]
title_list = [song_info_data.findInfobyID(sk).title for sk in song_keys]
album_list = [song_info_data.findInfobyID(sk).album for sk in song_keys]

#TODO match each voc index with its voc word
voc_list = [voc_dict[1][e] for e in voc_keys]



#TODO visualization using bokeh
from bokeh.io import vform
from bokeh.plotting import figure, output_file, show, ColumnDataSource,hplot
from bokeh.models import HoverTool,CustomJS,Slider,BoxZoomTool, ResetTool,WheelZoomTool,PanTool,LassoSelectTool,Rect

map_radius = 0.7
start_size = 5
start_alpha = 0.3

source_song = ColumnDataSource(
        data=dict(
            x=embedding[voc_size:,0].tolist(),
            y=embedding[voc_size:,1].tolist(),
            size = [start_size]*(lyrics_size),
            title=title_list,
            album=album_list,
            artist=artist_list,
            alpha=[start_alpha]*(lyrics_size)
            )
    )
source_voc = ColumnDataSource(
        data=dict(
            x=embedding[:voc_size,0].tolist(),
            y=embedding[:voc_size,1].tolist(),
            size = [start_size]*(voc_size),#[start_size]*(voc_size),
            voc = voc_list,
            alpha=[start_alpha]*(voc_size),
            )
    )


source_voc_fix = ColumnDataSource(
    data=dict(
        size = [start_size],
        alpha = [start_alpha]
    ))
source_voc_size = ColumnDataSource(
    data=dict(
        size = [start_size]#[start_size]*(voc_size),
    ))

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

                voc_data['color'][i] = rgbToHex(255, 0, 0);
            }else{
                if(select_array[i]==1){
                    voc_data['alpha'][i] = voc_alpha;

                    voc_size_tmp[i]=voc_data['size'][i];
                }else{
                    voc_data['alpha'][i] = 0;

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
p_voc.circle('x', 'y', size='size',fill_alpha = 'alpha',fill_color="#%02x%02x%02x"%(255, 0, 0),line_color=None, source=source_voc)


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
            //voc_size[i]=Math.pow(voc_size_tmp[i],f);
            voc_size[i]=f;
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

p_map.scatter(x=embedding[:,0].tolist(), y=embedding[:,1].tolist(), radius=map_radius, fill_alpha=0.08,color=colors, line_color=None)

rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.3,
            line_color='black', fill_color='black')
p_map.add_glyph(source, rect)
output_file(outputfilename+".html")
show(vform(hplot(p_map,vform(slider_alpha,slider_size)),hplot(p_voc,p_song)))
