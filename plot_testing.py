from bokeh.io import vform
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool,CustomJS,Slider,BoxZoomTool, ResetTool,WheelZoomTool,PanTool
import numpy as np
output_file("toolbar.html")

lyrics_size = len(lyrics_data.ids)
voc_size = len(lyrics_data.voc_dict[0])
lyrics_size=14883
voc_size=90218

colors_red = ["#%02x%02x%02x"%(int(r), int(g), 0) for r, g in zip([255]*voc_size, [0]*voc_size)]
colors_green = ["#%02x%02x%02x"%(int(r), int(g), 0) for r, g in zip([0]*lyrics_size, [255]*lyrics_size)]
colors = []
colors.extend(colors_red)
colors.extend(colors_green)

voc_list = [lyrics_data.voc_dict[1][i] for i in range(voc_size)]
len(voc_list)

sorted_ids = np.sort(lyrics_data.ids)

title_list = [song_info_data.findInfobyID(element).title for element in sorted_ids]
album_list = [song_info_data.findInfobyID(element).album for element in sorted_ids]
artist_list = [song_info_data.findInfobyID(element).artist for element in sorted_ids]

#XXX extend the label list to full size
voc_list.extend([""]*lyrics_size)
title_list = [""]*voc_size+title_list
album_list = [""]*voc_size+album_list
artist_list = [""]*voc_size+artist_list
len(voc_list)
len(title_list)
len(album_list)
len(artist_list)

#REVIEW restart here!
source = ColumnDataSource(
        data=dict(
            x=embedding_matrix[:,0],
            y=embedding_matrix[:,1],
            size = [1]*(lyrics_size+voc_size),
            title=title_list,
            album=album_list,
            artist=artist_list,
            voc = voc_list,
            alpha=[0.1]*(lyrics_size+voc_size),
            color=colors
            )
    )

hover = HoverTool(
        tooltips=[
            ("title", "@title"),
            ("album", "@album"),
            ("artist", "@artist"),
            ("voc", "@voc")
        ]
    )
Tools = [BoxZoomTool(), ResetTool(),WheelZoomTool(),PanTool()]
Tools.append(hover)
p = figure(tools=Tools,
           title="Testing",webgl=True)

p.circle('x', 'y', size='size',fill_alpha = 'alpha',fill_color='color',line_color=None, source=source)

callback_alpha = CustomJS(args=dict(source=source), code="""
        var data = source.get('data');
        var f = cb_obj.get('value');
        alpha = data['alpha'];
        for (i=0;i<alpha.length;i++){
            alpha[i] = f;
        }
        source.trigger('change');
    """)
callback_size = CustomJS(args=dict(source=source), code="""
        var data = source.get('data');
        var f = cb_obj.get('value');
        size = data['size'];
        for (i=0;i<size.length;i++){
            size[i] = f;
        }
        source.trigger('change');
    """)
##########################################################
slider_alpha = Slider(start=0.0, end=1, value=0.1, step=.01, title="fill_alpha", callback=callback_alpha)
slider_size = Slider(start=0.1, end=10, value=1, step=.1, title="size", callback=callback_size)

show(vform(vform(slider_alpha,slider_size),p))
