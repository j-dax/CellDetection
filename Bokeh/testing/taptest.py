from bokeh.models import ColumnDataSource, CustomJS, Tap
from bokeh.plotting import figure, output_file, show

output_file("openurl.html")

p = figure(plot_width=400, plot_height=400,
             tools="tap", title="Click the Dots")

source = ColumnDataSource(data=dict(x=[1, 2, 3, 4, 5, 5], y=[2, 5, 8, 2, 7, 7]))

p.square('x', 'y', color='green', size=20, source=source)

taptool = p.select(type=Tap)
callback = CustomJS(args=dict(), code="""
    console.log(cb_obj.x);
    console.log(cb_obj.sx);
""")

# source.js_on_change()
p.js_on_event('tap', callback)
show(p)