'''from bokeh.plotting import figure, show ,output_file
from bokeh.models import HoverTool, ColorBar, SaveTool, NumeralTickFormatter, LinearColorMapper, LassoSelectTool, ResetTool, PanTool, BoxSelectTool, TapTool, PolySelectTool


output_file('image.html')
p = figure(x_range=(0,1), y_range=(0,1))
p.add_tools(LassoSelectTool())

show(p)'''

import random
from bokeh.layouts import row 
from bokeh.plotting import figure, output_file, show
from bokeh.models import CustomJS, Div, Row, SaveTool, LassoSelectTool, ColumnDataSource


output_file("callback.html")

x = [random.random() for x in range(500)]
y = [random.random() for y in range(500)]
s1 = ColumnDataSource(data=dict(x=x, y=y))

p1 = figure(plot_width = 500, plot_height = 500, x_range=(0,1), y_range=(0,1),
    tools="lasso_select", title = "select here")
url = "https://previews.123rf.com/images/tonaquatic19/tonaquatic191903/tonaquatic19190300060/120853028-human-lung-tissue-under-microscope-view-lungs-are-the-primary-organs-of-the-respiratory-system-in-hu.jpg"
d1 = Div(text = '<div style="position: absolute; left:-475px; top:25px"><img src=' + 
    url + ' style="width:400px; height:400px; opacity: 0.3"></div>')

p1.image_url(url=[url], x=0, y=1, w=0.8, h=0.6)
p1.circle('x', 'y', source=s1, alpha=0.6)
p1.add_tools(SaveTool())

s2 = ColumnDataSource(data=dict(x=[], y=[]))
p2 = figure(plot_width = 500, plot_height = 500, x_range=(0,1),
    y_range=(0,1), tools="lasso_select", title = "watch here")
p2.circle('x', 'y', source=s2, alpha=0.6)
p2.add_tools(SaveTool())

s1.selected.js_on_change('indices', CustomJS(args=dict(s1=s1, s2=s2), code="""
        var inds = cb_obj.indices;
        var d1 = s1.data;
        var d2 = s2.data;
        d2['x'] = []
        d2['y'] = []
        for (var i = 0; i < inds.length; i++) {
            d2['x'].push(d1['x'][inds[i]])
            d2['y'].push(d1['y'][inds[i]])
        }
        s2.change.emit();
    """)
)

layout = row(Row(p1, d1), p2)
show(layout)