from bokeh.io import vform
from bokeh.plotting import figure, output_file, show, ColumnDataSource,hplot
from bokeh.models import HoverTool,CustomJS,Slider,BoxZoomTool, ResetTool,WheelZoomTool,PanTool,LassoSelectTool
import numpy as np
output_file("toolbar.html")

lyrics_size = len(lyrics_data.ids)
voc_size = len(lyrics_data.voc_dict[0])
lyrics_size=14883
voc_size=90218

voc_list = [lyrics_data.voc_dict[1][i] for i in range(voc_size)]
len(voc_list)

sorted_ids = np.sort(lyrics_data.ids)

title_list = [song_info_data.findInfobyID(element).title for element in sorted_ids]
album_list = [song_info_data.findInfobyID(element).album for element in sorted_ids]
artist_list = [song_info_data.findInfobyID(element).artist for element in sorted_ids]

#XXX extend the label list to full size

len(voc_list)
len(title_list)
len(album_list)
len(artist_list)

#REVIEW restart here!
source_song = ColumnDataSource(
        data=dict(
            x=embedding_matrix[voc_size:,0],
            y=embedding_matrix[voc_size:,1],
            size = [1]*(lyrics_size),
            title=title_list,
            album=album_list,
            artist=artist_list,
            #voc = voc_list,
            alpha=[0.1]*(lyrics_size)
            #color=colors,
            #SorV = [0]*(voc_size)+[1]*(lyrics_size)
            )
    )
source_voc = ColumnDataSource(
        data=dict(
            x=embedding_matrix[:voc_size,0],
            y=embedding_matrix[:voc_size,1],
            size = [1]*(voc_size),
            #title=title_list,
            #album=album_list,
            #artist=artist_list,
            voc = voc_list,
            alpha=[0.1]*(voc_size),
            #color=colors,
            #SorV = [1]*(voc_size)+[0]*(lyrics_size)
            )
    )
hover_song = HoverTool(
        tooltips=[
            ("title", "@title"),
            ("album", "@album"),
            ("artist", "@artist")        ]
    )
hover_voc = HoverTool(
        tooltips=[
            ("voc", "@voc")
        ]
    )
Tools_song = [BoxZoomTool(), ResetTool(),WheelZoomTool(),PanTool(),hover_song,LassoSelectTool()]
Tools_voc = [BoxZoomTool(), ResetTool(),WheelZoomTool(),PanTool(),hover_voc,LassoSelectTool()]
p_song = figure(tools=Tools_song,
           title="Testing",webgl=True)
p_voc = figure(tools=Tools_voc,
           title="Testing",webgl=True,x_range = p_song.x_range,y_range = p_song.y_range)
p_song.circle('x', 'y', size='size',fill_alpha = 'alpha',fill_color="#%02x%02x%02x"%(0, 255, 0),line_color=None, source=source_song)
p_voc.circle('x', 'y', size='size',fill_alpha = 'alpha',fill_color="#%02x%02x%02x"%(255, 0, 0),line_color=None, source=source_voc)


callback_alpha = CustomJS(args=dict(source_song=source_song,source_voc=source_voc), code="""
        var song_data = source_song.get('data');
        var voc_data = source_voc.get('data');
        var f = cb_obj.get('value');
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
callback_size = CustomJS(args=dict(source_song=source_song,source_voc=source_voc), code="""
        var song_data = source_song.get('data');
        var voc_data = source_voc.get('data');
        var f = cb_obj.get('value');
        song_size = song_data['size'];
        voc_size = voc_data['size'];
        for (i=0;i<song_size.length;i++){
            song_size[i]=f;
        }
        for (i=0;i<voc_size.length;i++){
            voc_size[i]=f;
        }
        source_song.trigger('change');
        source_voc.trigger('change');
    """)
##########################################################
slider_alpha = Slider(start=0.1, end=1, value=0.3, step=.1, title="fill_alpha", callback=callback_alpha)
slider_size = Slider(start=1, end=10, value=5, step=1, title="size", callback=callback_size)

show(vform(vform(slider_alpha,slider_size),hplot(p_song,p_voc)))
